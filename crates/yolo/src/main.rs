use std::path::PathBuf;

use image::{GrayImage, Luma, imageops};
use ultralytics_inference::{Device, InferenceConfig, YOLOModel, annotate::load_image};

const BACKGROUND_SIGMA: f32 = 15.0;
const MASK_FEATHER_SIGMA: f32 = 2.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::CoreMl;
    let config = InferenceConfig::new()
        .with_confidence(0.25)
        .with_iou(0.45)
        .with_rect(false)
        .with_device(device);

    let mut model = YOLOModel::load_with_config("yolo26m-seg.onnx", config)?;
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("input")
        .join("pessoas_felizes.jpg");
    let results = model.predict(base.clone())?;
    let image = load_image(&base.to_string_lossy())?;

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resultados")
        .join("pessoas_felizes_blur.png");
    for result in results {
        let Some(masks) = result.masks.as_ref() else {
            continue;
        };

        let sharp = image.to_rgb8();
        let (width, height) = sharp.dimensions();

        // União de todas as instâncias detectadas = primeiro plano.
        let mut foreground = GrayImage::new(width, height);
        for plane in masks.data.outer_iter() {
            for y in 0..height {
                for x in 0..width {
                    if plane[[y as usize, x as usize]] > 0.5 {
                        foreground.put_pixel(x, y, Luma([255]));
                    }
                }
            }
        }
        // Borda suave para não deixar serrilhado entre pessoa e fundo.
        let foreground = imageops::blur(&foreground, MASK_FEATHER_SIGMA);

        let blurred = imageops::blur(&sharp, BACKGROUND_SIGMA);
        let mut output = blurred.clone();
        for (x, y, px) in output.enumerate_pixels_mut() {
            let alpha = f32::from(foreground.get_pixel(x, y).0[0]) / 255.0;
            let fg = sharp.get_pixel(x, y).0;
            let bg = blurred.get_pixel(x, y).0;
            for c in 0..3 {
                px.0[c] = (f32::from(fg[c]) * alpha + f32::from(bg[c]) * (1.0 - alpha)) as u8;
            }
        }
        output.save(&output_path)?;
    }

    Ok(())
}
