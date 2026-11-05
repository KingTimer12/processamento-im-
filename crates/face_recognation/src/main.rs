use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use image::{DynamicImage, RgbImage};
use opencv::{
    Result,
    core::{self, AlgorithmHint::ALGO_HINT_DEFAULT},
    highgui, imgproc,
    objdetect::{self, FaceDetectorYN, FaceRecognizerSF},
    prelude::*,
    videoio,
};
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

const WINDOW: &str = "Reconhecimento facial";
// Limiar recomendado pelo SFace para distância cosseno.
const MATCH_THRESHOLD: f64 = 0.363;
// Abaixo disso o rosto é considerado de outra pessoa; entre os dois limiares fica em dúvida
// e não pede cadastro.
const UNKNOWN_THRESHOLD: f64 = 0.25;
// Reconhecimentos com score nessa faixa viram amostras novas da pessoa (outros ângulos/distâncias).
const SAMPLE_MIN_SCORE: f64 = 0.40;
const SAMPLE_MAX_SCORE: f64 = 0.60;
const MAX_SAMPLES: usize = 20;
// Só pede cadastro de rostos nítidos e longe da borda do frame.
const REGISTER_MIN_CONF: f32 = 0.6;
const REGISTER_MIN_SIZE: i32 = 80;
const BORDER_MARGIN: i32 = 15;
// Quantos frames seguidos o rosto precisa ficar desconhecido antes de pedir o nome.
const UNKNOWN_FRAMES: u32 = 15;
// Frames de espera após um cadastro cancelado, para não perguntar de novo em seguida.
const PROMPT_COOLDOWN: u32 = 60;
// Margem ao redor da caixa do YOLO e tamanho mínimo do recorte para o YuNet achar os landmarks.
const CROP_MARGIN: f32 = 0.3;
const CROP_MIN_SIZE: f64 = 224.0;

struct Person {
    name: String,
    features: Vec<Mat>,
}

fn to_cv(e: impl std::fmt::Display) -> opencv::Error {
    opencv::Error::new(core::StsError, e.to_string())
}

fn load_people(path: &Path) -> Result<Vec<Person>> {
    let Ok(text) = fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    let mut people: Vec<Person> = Vec::new();
    for line in text.lines() {
        let Some((name, values)) = line.split_once(';') else {
            continue;
        };
        let values: Vec<f32> = values
            .split_whitespace()
            .filter_map(|v| v.parse().ok())
            .collect();
        let feature = Mat::from_slice(&values)?.try_clone()?;
        match people.iter_mut().find(|p| p.name == name) {
            Some(person) => person.features.push(feature),
            None => people.push(Person {
                name: name.to_string(),
                features: vec![feature],
            }),
        }
    }
    Ok(people)
}

fn save_sample(path: &Path, name: &str, feature: &Mat) -> Result<()> {
    let values: Vec<String> = feature
        .data_typed::<f32>()?
        .iter()
        .map(f32::to_string)
        .collect();
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(to_cv)?;
    writeln!(file, "{};{}", name, values.join(" ")).map_err(to_cv)
}

/// Retorna o índice da pessoa mais parecida e o score (melhor entre todas as amostras dela).
fn best_match(
    recognizer: &core::Ptr<FaceRecognizerSF>,
    people: &[Person],
    feature: &Mat,
) -> Result<Option<(usize, f64)>> {
    let mut best: Option<(usize, f64)> = None;
    for (i, person) in people.iter().enumerate() {
        for sample in &person.features {
            let score = recognizer.match_(
                sample,
                feature,
                objdetect::FaceRecognizerSF_DisType::FR_COSINE as i32,
            )?;
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((i, score));
            }
        }
    }
    Ok(best)
}

/// Recorta a caixa do YOLO com margem, amplia rostos pequenos e alinha pelos 5 landmarks
/// do YuNet, que é o formato em que o SFace foi treinado. `None` se não achar os landmarks.
fn aligned_face(
    frame: &Mat,
    rect: core::Rect,
    landmarks: &mut core::Ptr<FaceDetectorYN>,
    recognizer: &core::Ptr<FaceRecognizerSF>,
) -> Result<Option<Mat>> {
    let size = frame.size()?;
    let dx = (rect.width as f32 * CROP_MARGIN) as i32;
    let dy = (rect.height as f32 * CROP_MARGIN) as i32;
    let padded = core::Rect::new(
        rect.x - dx,
        rect.y - dy,
        rect.width + 2 * dx,
        rect.height + 2 * dy,
    ) & core::Rect::new(0, 0, size.width, size.height);
    let crop = Mat::roi(frame, padded)?;

    let scale = (CROP_MIN_SIZE / f64::from(padded.width.min(padded.height))).max(1.0);
    let mut crop_scaled = Mat::default();
    imgproc::resize(
        &crop,
        &mut crop_scaled,
        core::Size::default(),
        scale,
        scale,
        imgproc::INTER_CUBIC,
    )?;

    landmarks.set_input_size(crop_scaled.size()?)?;
    let mut detections = Mat::default();
    landmarks.detect(&crop_scaled, &mut detections)?;
    // Cada linha: [x, y, w, h, 5 landmarks (x, y), score]; fica a de maior score.
    let best = (0..detections.rows())
        .max_by(|&a, &b| {
            let sa = *detections.at_2d::<f32>(a, 14).unwrap_or(&0.0);
            let sb = *detections.at_2d::<f32>(b, 14).unwrap_or(&0.0);
            sa.total_cmp(&sb)
        });
    let Some(best) = best else {
        return Ok(None);
    };

    let mut aligned = Mat::default();
    recognizer.align_crop(&crop_scaled, &detections.row(best)?, &mut aligned)?;
    Ok(Some(aligned))
}

fn ask_name() -> Option<String> {
    print!("Rosto desconhecido. Digite o nome (vazio para ignorar): ");
    io::stdout().flush().ok()?;
    let mut name = String::new();
    io::stdin().read_line(&mut name).ok()?;
    let name = name.trim().replace(';', "");
    (!name.is_empty()).then_some(name)
}

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models = root.join("../..");
    let config = InferenceConfig::new()
        .with_confidence(0.4)
        .with_iou(0.45)
        .with_device(Device::CoreMl);
    let mut detector =
        YOLOModel::load_with_config(models.join("face.onnx"), config).map_err(to_cv)?;
    let mut landmarks = FaceDetectorYN::create(
        &models
            .join("face_detection_yunet_2023mar.onnx")
            .to_string_lossy(),
        "",
        core::Size::new(320, 320),
        0.5,
        0.3,
        10,
        0,
        0,
    )?;
    let mut recognizer = FaceRecognizerSF::create(
        &models
            .join("face_recognition_sface_2021dec.onnx")
            .to_string_lossy(),
        "",
        0,
        0,
    )?;

    let db_path = root.join("rostos.txt");
    let mut people = load_people(&db_path)?;
    println!("{} pessoa(s) cadastrada(s)", people.len());

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Error: Unable to open the webcam.");
    }
    highgui::named_window(WINDOW, highgui::WINDOW_AUTOSIZE)?;
    println!("'q' ou ESC para sair");

    let mut frame = Mat::default();
    let mut rgb = Mat::default();
    let mut feature = Mat::default();
    let mut cooldown = 0;
    let mut unknown_streak = 0;
    loop {
        cam.read(&mut frame)?;
        if frame.empty() {
            continue;
        }
        cooldown = u32::saturating_sub(cooldown, 1);

        imgproc::cvt_color(&frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0, ALGO_HINT_DEFAULT)?;
        let size = rgb.size()?;
        let (w, h) = (size.width as u32, size.height as u32);
        let buffer = RgbImage::from_raw(w, h, rgb.data_bytes()?.to_vec())
            .expect("buffer RGB com tamanho inválido");

        let results = detector
            .predict_image(&DynamicImage::ImageRgb8(buffer), "webcam".to_string())
            .map_err(to_cv)?;

        // Recorta e reconhece antes de desenhar, para o recorte não pegar as anotações.
        let bounds = core::Rect::new(0, 0, size.width, size.height);
        let inner = core::Rect::new(
            BORDER_MARGIN,
            BORDER_MARGIN,
            size.width - 2 * BORDER_MARGIN,
            size.height - 2 * BORDER_MARGIN,
        );
        let mut unknown = None;
        let mut labels = Vec::new();
        for boxes in results.iter().filter_map(|r| r.boxes.as_ref()) {
            for row in boxes.data.outer_iter() {
                let rect = core::Rect::new(
                    row[0] as i32,
                    row[1] as i32,
                    (row[2] - row[0]) as i32,
                    (row[3] - row[1]) as i32,
                ) & bounds;
                if rect.width <= 0 || rect.height <= 0 {
                    continue;
                }

                let Some(aligned) = aligned_face(&frame, rect, &mut landmarks, &recognizer)? else {
                    labels.push((rect, "?".to_string(), core::Scalar::new(0.0, 255.0, 255.0, 0.0)));
                    continue;
                };
                recognizer.feature(&aligned, &mut feature)?;

                let good_quality = row[4] >= REGISTER_MIN_CONF
                    && rect.width.min(rect.height) >= REGISTER_MIN_SIZE
                    && (rect & inner) == rect;
                match best_match(&recognizer, &people, &feature)? {
                    Some((i, score)) if score >= MATCH_THRESHOLD => {
                        let person = &mut people[i];
                        if good_quality
                            && (SAMPLE_MIN_SCORE..SAMPLE_MAX_SCORE).contains(&score)
                            && person.features.len() < MAX_SAMPLES
                        {
                            save_sample(&db_path, &person.name, &feature)?;
                            person.features.push(feature.try_clone()?);
                        }
                        labels.push((
                            rect,
                            format!("{} {:.2}", person.name, score),
                            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                        ));
                    }
                    Some((_, score)) if score >= UNKNOWN_THRESHOLD => {
                        labels.push((rect, format!("? {score:.2}"), core::Scalar::new(0.0, 255.0, 255.0, 0.0)));
                    }
                    _ => {
                        if good_quality && unknown.is_none() {
                            unknown = Some(feature.try_clone()?);
                        }
                        labels.push((rect, "desconhecido".to_string(), core::Scalar::new(0.0, 0.0, 255.0, 0.0)));
                    }
                }
            }
        }
        unknown_streak = if unknown.is_some() { unknown_streak + 1 } else { 0 };

        for (rect, label, color) in &labels {
            imgproc::rectangle(&mut frame, *rect, *color, 2, imgproc::LINE_8, 0)?;
            imgproc::put_text(
                &mut frame,
                label,
                core::Point::new(rect.x, (rect.y - 8).max(12)),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                *color,
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        highgui::imshow(WINDOW, &frame)?;
        let key = highgui::wait_key(1)?;
        if key == i32::from(b'q') || key == 27 {
            break;
        }

        // O vídeo congela enquanto o console espera o nome.
        if let Some(feature) = unknown.filter(|_| cooldown == 0 && unknown_streak >= UNKNOWN_FRAMES) {
            unknown_streak = 0;
            match ask_name() {
                Some(name) => {
                    save_sample(&db_path, &name, &feature)?;
                    println!("'{name}' cadastrado");
                    match people.iter_mut().find(|p| p.name == name) {
                        Some(person) => person.features.push(feature),
                        None => people.push(Person { name, features: vec![feature] }),
                    }
                }
                None => cooldown = PROMPT_COOLDOWN,
            }
        }
    }
    Ok(())
}
