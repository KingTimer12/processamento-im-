//! Laboratorio guiado - Restauracao fotografica (versao Rust + OpenCV).
//!
//! Pipeline: mediana -> inpainting -> CLAHE -> Unsharp Mask -> painel antes/depois.

use std::fs;
use std::path::PathBuf;

use opencv::{
    core::{self, AlgorithmHint, BORDER_DEFAULT, Mat, Size, Vector},
    imgcodecs::{self, IMREAD_GRAYSCALE},
    imgproc, photo,
    prelude::*,
};

fn salvar(caminho: &PathBuf, img: &Mat) -> opencv::Result<()> {
    imgcodecs::imwrite(caminho.to_str().unwrap(), img, &Vector::new())?;
    println!("Gravado: {}", caminho.display());
    Ok(())
}

/// Equivalente ao `ImageFilter.UnsharpMask(radius, percent, threshold)` do Pillow:
/// desfoca com gaussiana de desvio `radius` e soma `percent`% da diferenca
/// onde ela for maior ou igual a `threshold`.
fn unsharp_mask(img: &Mat, radius: f64, percent: i32, threshold: i32) -> opencv::Result<Mat> {
    let mut borrada = Mat::default();
    imgproc::gaussian_blur(img, &mut borrada, Size::new(0, 0), radius, radius, BORDER_DEFAULT, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let mut saida = img.try_clone()?;
    let origem = img.data_bytes()?;
    let suave = borrada.data_bytes()?;
    for (i, px) in saida.data_bytes_mut()?.iter_mut().enumerate() {
        let diferenca = origem[i] as i32 - suave[i] as i32;
        if diferenca.abs() >= threshold {
            *px = (origem[i] as i32 + diferenca * percent / 100).clamp(0, 255) as u8;
        }
    }
    Ok(saida)
}

fn main() -> opencv::Result<()> {
    // ETAPA 1: preparar as pastas e abrir os arquivos.
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("input");
    let saida = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resultados");
    fs::create_dir_all(&saida).expect("nao foi possivel criar a pasta resultados");

    let imagem = imgcodecs::imread(base.join("foto_degradada.png").to_str().unwrap(), IMREAD_GRAYSCALE)?;
    let mascara = imgcodecs::imread(base.join("mascara_riscos.png").to_str().unwrap(), IMREAD_GRAYSCALE)?;
    if imagem.empty() || mascara.empty() {
        panic!("Confira se os dois arquivos PNG estao na pasta do projeto.");
    }
    if imagem.size()? != mascara.size()? {
        panic!("A fotografia e a mascara precisam ter as mesmas dimensoes.");
    }

    println!("Fotografia carregada: {} x {} pixels", imagem.cols(), imagem.rows());
    println!("Mascara carregada. Pixels brancos indicam areas com riscos.");

    // ETAPA 2: remover ruido pontual com mediana 3x3.
    let mut sem_ruido = Mat::default();
    imgproc::median_blur(&imagem, &mut sem_ruido, 3)?;
    salvar(&saida.join("01_sem_ruido.png"), &sem_ruido)?;

    // ETAPA 3: reconstruir os riscos marcados na mascara (inpainting Telea, raio 3).
    let mut sem_riscos = Mat::default();
    photo::inpaint(&sem_ruido, &mascara, &mut sem_riscos, 3.0, photo::INPAINT_TELEA)?;
    salvar(&saida.join("02_sem_riscos.png"), &sem_riscos)?;

    // ETAPA 4: contraste local com CLAHE.
    let mut clahe = imgproc::create_clahe(1.8, Size::new(8, 8))?;
    let mut contraste = Mat::default();
    clahe.apply(&sem_riscos, &mut contraste)?;
    salvar(&saida.join("03_contraste.png"), &contraste)?;

    // ETAPA 5: nitidez com Unsharp Mask.
    let restaurada = unsharp_mask(&contraste, 1.1, 95, 2)?;
    salvar(&saida.join("04_restaurada.png"), &restaurada)?;

    // ETAPA 6: painel ANTES / DEPOIS lado a lado.
    let mut lados = Vector::<Mat>::new();
    lados.push(imagem);
    lados.push(restaurada);
    let mut painel_cinza = Mat::default();
    core::hconcat(&lados, &mut painel_cinza)?;
    let mut painel = Mat::default();
    imgproc::cvt_color_def(&painel_cinza, &mut painel, imgproc::COLOR_GRAY2BGR)?;
    salvar(&saida.join("05_antes_depois.png"), &painel)?;

    Ok(())
}
