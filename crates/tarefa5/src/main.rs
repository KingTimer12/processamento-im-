use opencv::{
    core::{CV_8UC3, Point, Scalar, Size, Vector},
    imgcodecs,
    imgproc::{self, FONT_HERSHEY_SIMPLEX, LINE_8},
};

fn main() -> opencv::Result<()> {
    let (largura, altura) = (800, 500);
    let mut img = opencv::core::Mat::new_size_with_default(
        Size::new(largura, altura),
        CV_8UC3,
        Scalar::new(235.0, 242.0, 248.0, 0.0),
    )?;

    let (rect_w, rect_h) = (220, 140);

    let rect = opencv::core::Rect_ {
        x: 100,
        y: 180,
        width: rect_w,
        height: rect_h,
    };
    let color = Scalar::new(255.0, 255.0, 255.0, 0.0);
    imgproc::rectangle(&mut img, rect, color, -1, LINE_8, 0)?;

    let center = Point::new(100 + (rect_w/2), 180 + (rect_h/2));
    let color = Scalar::new(0.0, 0.0, 255.0, 0.0);
    imgproc::circle(&mut img, center, 40, color, -1, LINE_8, 0)?;

    // let color = Scalar::all(0.0);
    // imgproc::line(
    //     &mut img,
    //     Point::new(60, 380),
    //     Point::new(730, 380),
    //     color,
    //     1,
    //     LINE_8,
    //     0,
    // )?;

    let color = Scalar::all(0.0);
    imgproc::put_text(
        &mut img,
        "Japão",
        Point::new(70 + (rect_w/2), 160),
        FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        1,
        LINE_8,
        false,
    )?;

    imgcodecs::imwrite("./img.png", &img, &Vector::new())?;

    Ok(())
}
