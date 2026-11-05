use opencv::{
    Result,
    core::{self, AlgorithmHint::ALGO_HINT_DEFAULT, Mat, Rect, Size, Vector},
    imgcodecs, imgproc, photo,
    prelude::*,
};

/// Região (x, y, largura, altura) da placa no caso_1.jpeg original, com margem.
const PLACA_CASO_1: Rect = Rect {
    x: 200,
    y: 428,
    width: 92,
    height: 46,
};
const ESCALA: f64 = 8.0;

/// Isola a placa do caso_1 e aplica realce com OpenCV, salvando cada etapa.
pub fn revelar_placa_caso_1() -> Result<()> {
    let imagem = imgcodecs::imread("evidencias/caso_1.jpeg", imgcodecs::IMREAD_COLOR)?;
    let recorte = Mat::roi(&imagem, PLACA_CASO_1)?.try_clone()?;

    // Remove o ruído de sensor (câmera noturna) preservando bordas, ainda em cores
    let mut sem_ruido = Mat::default();
    photo::fast_nl_means_denoising_colored(&recorte, &mut sem_ruido, 4.0, 4.0, 5, 15)?;

    let mut cinza = Mat::default();
    imgproc::cvt_color(
        &sem_ruido,
        &mut cinza,
        imgproc::COLOR_BGR2GRAY,
        0,
        ALGO_HINT_DEFAULT,
    )?;

    // Amplia com interpolação bicúbica para os caracteres terem pixels suficientes
    let mut ampliada = Mat::default();
    imgproc::resize(
        &cinza,
        &mut ampliada,
        Size::default(),
        ESCALA,
        ESCALA,
        imgproc::INTER_CUBIC,
    )?;

    // Equalização adaptativa: realça contraste local sem estourar o reflexo da placa
    let mut clahe = imgproc::create_clahe(3.0, Size::new(4, 4))?;
    let mut equalizada = Mat::default();
    clahe.apply(&ampliada, &mut equalizada)?;

    // Suaviza o serrilhado do upscale e reforça as bordas (unsharp mask)
    let mut suave = Mat::default();
    imgproc::bilateral_filter(&equalizada, &mut suave, 9, 50.0, 50.0, core::BORDER_DEFAULT)?;
    let mut borrada = Mat::default();
    imgproc::gaussian_blur(
        &suave,
        &mut borrada,
        Size::new(0, 0),
        5.0,
        0.0,
        core::BORDER_DEFAULT,
        ALGO_HINT_DEFAULT,
    )?;
    let mut nitida = Mat::default();
    core::add_weighted(&suave, 1.8, &borrada, -0.8, 0.0, &mut nitida, -1)?;

    // Binariza com limiar local: caracteres escuros sobre placa clara,
    // mesmo com iluminação desigual (farol de um lado, sombra do outro)
    let mut binaria = Mat::default();
    imgproc::adaptive_threshold(
        &nitida,
        &mut binaria,
        255.0,
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY,
        51,
        5.0,
    )?;

    salvar("evidencias/caso_1_cv_1_recorte.png", &recorte)?;
    salvar("evidencias/caso_1_cv_2_sem_ruido.png", &sem_ruido)?;
    salvar("evidencias/caso_1_cv_3_equalizada.png", &equalizada)?;
    salvar("evidencias/caso_1_cv_4_nitida.png", &nitida)?;
    salvar("evidencias/caso_1_cv_5_binaria.png", &binaria)?;

    Ok(())
}

fn salvar(path: &str, mat: &Mat) -> Result<()> {
    imgcodecs::imwrite(path, mat, &Vector::new())?;
    Ok(())
}
