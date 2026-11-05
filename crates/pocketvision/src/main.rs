use opencv::{
    Result,
    core::{self, AlgorithmHint::ALGO_HINT_DEFAULT},
    geometry, highgui,
    imgproc::{self, COLOR_BGR2HSV},
    prelude::*,
    videoio,
};

const MIN_AREA: f64 = 1000.0;

fn main() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Error: Unable to open the webcam.");
    }
    let window_name = "Rust OpenCV Webcam Feed";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let mut frame = core::Mat::default();

    println!("Controles:");
    println!("  'h' -> Ativar visão HSV");
    println!("  'm' -> Ativar Máscara (Filtro de Objeto AZUL)");
    println!("  'c' -> Ativar Contorno do objeto azul");
    println!("  'n' -> Resetar tudo para o normal");
    println!("  'q' ou 'ESC' -> Sair");

    let mut status = vec![];

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(5, 5),
        core::Point::new(-1, -1),
    )?;

    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            continue;
        }

        let mut dest = frame.clone();
        let mut final_mask = core::Mat::default();

        let key = highgui::wait_key(10)?;
        if key == 'h' as i32 { status.push("HSV"); println!("Modo: HSV"); }
        if key == 'm' as i32 { status.push("MASK"); println!("Modo: MASK"); }
        if key == 'c' as i32 { status.push("COUNTOUR"); println!("Modo: COUNTOUR"); }
        if key == 'n' as i32 { status.clear(); println!("Modo: NORMAL"); }

        let mut hsv = core::Mat::default();
        if status.contains(&"HSV") {
            imgproc::cvt_color(&frame, &mut hsv, COLOR_BGR2HSV, 0, ALGO_HINT_DEFAULT)?;
            dest = hsv.clone();
        }

        if status.contains(&"MASK") {
            let mut mask = core::Mat::default();
            let mut processed_mask = core::Mat::default();
            
            let lower = core::Scalar::new(90.0, 7.0, 50.0, 0.0);
            let upper = core::Scalar::new(130.0, 255.0, 255.0, 0.0);
            core::in_range(&hsv, &lower, &upper, &mut mask)?;

            imgproc::morphology_ex(
                &mask,
                &mut processed_mask,
                imgproc::MORPH_OPEN,
                &kernel,
                core::Point::new(-1, -1),
                1,
                core::BORDER_CONSTANT,
                core::Scalar::default(),
            )?;

            imgproc::morphology_ex(
                &processed_mask,
                &mut final_mask,
                imgproc::MORPH_CLOSE,
                &kernel,
                core::Point::new(-1, -1),
                1,
                core::BORDER_CONSTANT,
                core::Scalar::default(),
            )?;

            dest = final_mask.clone();
        }

        if status.contains(&"COUNTOUR") {
            dest = frame.clone();
            let mut contours = core::Vector::<core::Vector<core::Point>>::new();
            imgproc::find_contours(
                &final_mask,
                &mut contours,
                imgproc::RETR_EXTERNAL,
                imgproc::CHAIN_APPROX_SIMPLE,
                core::Point::new(0, 0),
            )?;

            let mut target: Option<core::Vector<core::Point>> = None;
            let mut max_area = 0.0;

            for contour in contours.iter() {
                let area = geometry::contour_area(&contour, false)?;
                if area > max_area {
                    max_area = area;
                    target = Some(contour);
                }
            }

            if let Some(target_contour) = target {
                if max_area > MIN_AREA {
                    let mut target_vector = core::Vector::<core::Vector<core::Point>>::new();
                    target_vector.push(target_contour);
                    
                    imgproc::draw_contours(
                        &mut dest,
                        &target_vector,
                        0,
                        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                        3,
                        imgproc::LINE_8,
                        &core::no_array(),
                        i32::MAX,
                        core::Point::new(0, 0),
                    )?;
                }
            }
        }

        if frame.size()?.width > 0 && status.is_empty() {
            highgui::imshow(window_name, &frame)?;
        } else if dest.size()?.width > 0 {
            highgui::imshow(window_name, &dest)?;
        }

        if key == 'q' as i32 || key == 27 {
            break;
        }
    }

    highgui::destroy_window(window_name)?;
    Ok(())
}
