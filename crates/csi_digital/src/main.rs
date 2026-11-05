mod placa_opencv;

use std::error::Error;

use image::{ColorType, GenericImageView};
use pillow_rs::{Image, imageops_grayscale};

const OPERACOES: [&str; 4] = ["contraste", "mediana", "bordas", "nitidez"];

fn main() -> Result<(), Box<dyn Error>> {
    let image = Image::open("evidencias/evidencia_01.jpeg")?;

    for op in OPERACOES.iter() {
        let img_clone = aplicar_operacao(&image, op)?;
        let path = format!("evidencias/evidencia_01_{}.jpeg", op);
        img_clone.save(&path)?;
    }

    caso_1()?;
    placa_opencv::revelar_placa_caso_1()?;

    Ok(())
}

fn caso_1() -> Result<(), Box<dyn Error>> {
    let imagem = Image::open("evidencias/caso_1.jpeg")?;
    let foto = imagem.crop(Some((15, 280, 497, 675)))?;
    let resultado = imageops_grayscale(&foto)?
        .enhance_sharpness(4.0)?
        .enhance_contrast(2.0)?
        .gaussian_blur(1.5)?
        .median_filter(1)?;

    resultado.save("evidencias/caso_1_resultado.jpeg")?;

    Ok(())
}

// fn aplicar_operacoes(image: &Image, ops: &[&str]) -> Result<Image, Box<dyn Error>> {
//     let mut resultado = image.clone();
//     for op in ops {
//         resultado = aplicar_operacao(&resultado, op)?;
//     }
//     Ok(resultado)
// }

fn aplicar_operacao(image: &Image, op: &str) -> Result<Image, Box<dyn Error>> {
    // Os métodos do pillow-rs retornam uma nova Image; não alteram a original
    let resultado = match op {
        // Aumentar o contraste da imagem
        "contraste" => image.enhance_contrast(3.0)?,
        // Aplicar filtro de mediana
        "mediana" => image.median_filter(5)?,
        // Detectar bordas na imagem (em tons de cinza)
        "bordas" => image
            .convert("L", None, None, None, None)?
            .filter("FIND_EDGES")?,
        // Aumentar a nitidez da imagem
        "nitidez" => image.enhance_sharpness(4.0)?,
        _ => image.clone(),
    };
    Ok(resultado)
}

trait ImageOutput {
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>>;
}

impl ImageOutput for Image {
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let buf = self.tobytes()?;
        let (width, height) = self.size()?;
        let color = match self.mode()?.as_str() {
            "L" => ColorType::L8,
            "RGB" => ColorType::Rgb8,
            outro => return Err(format!("modo não suportado: {outro}").into()),
        };
        image::save_buffer(path, &buf, width, height, color)?;
        Ok(())
    }
}

trait ImageInput {
    fn open(path: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;
}

impl ImageInput for Image {
    fn open(path: &str) -> Result<Self, Box<dyn Error>> {
        let img = image::open(path)?;
        Ok(Image::frombytes(
            "RGB",
            img.dimensions(),
            img.to_rgb8().as_raw(),
        )?)
    }
}
