use opengl::prelude::*;

const VERT: &str = r#"
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
uniform mat4 uProj;
uniform mat4 uView;
uniform mat4 uModel;
uniform mat3 uNormalMatrix;
out vec3 vNormal;
void main() {
    vNormal = uNormalMatrix * aNormal;
    gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);
}
"#;

const FRAG: &str = r#"
#version 330 core
in vec3 vNormal;
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    float d = max(dot(normalize(vNormal), normalize(vec3(0.5, 1.0, 0.8))), 0.0);
    FragColor = vec4(uColor * (0.15 + d), 1.0);
}
"#;

fn main() {
    run(Config::new("Formas", 900, 600), |gl| {
        let shader = gl.shader(VERT, FRAG);

        let mut star = MeshBuilder::new();
        let points: Vec<Vec3> = (0..10)
            .map(|i| {
                let r = if i % 2 == 0 { 1.0 } else { 0.45 };
                let a = i as f32 * std::f32::consts::TAU / 10.0;
                Vec3::new(r * a.cos(), r * a.sin(), 0.0)
            })
            .collect();
        for i in 0..10 {
            star.triangle(Vec3::ZERO, points[i], points[(i + 1) % 10]);
        }

        let mut shapes = vec![
            (gl.cube(), Vec3::new(0.2, 0.5, 0.9)),
            (gl.sphere(1.0, 32), Vec3::new(0.9, 0.3, 0.3)),
            (gl.pyramid(4, 1.2, 2.0), Vec3::new(0.9, 0.8, 0.2)),
            (gl.cylinder(0.8, 2.0, 24), Vec3::new(0.3, 0.8, 0.4)),
            (gl.cone(0.9, 2.0, 24), Vec3::new(0.7, 0.3, 0.9)),
            (gl.prism(6, 0.9, 1.6), Vec3::new(0.2, 0.8, 0.8)),
            (gl.build(&star), Vec3::new(1.0, 0.6, 0.1)),
        ];
        let n = shapes.len() as f32;
        for (i, (mesh, _)) in shapes.iter_mut().enumerate() {
            mesh.transform.position = Vec3::new((i as f32 - (n - 1.0) / 2.0) * 3.0, 0.0, 0.0);
        }

        move |gl, frame| {
            if frame.key_down(KeyCode::Escape) {
                frame.exit();
            }
            gl.clear(0.05, 0.05, 0.08);
            shader.set("uProj", perspective(45.0, frame.aspect(), 0.1, 100.0));
            shader.set("uView", look_at(Vec3::new(0.0, 4.0, 12.0), Vec3::ZERO));
            for (mesh, color) in shapes.iter_mut() {
                mesh.transform.spin(Vec3::Y, 50.0, frame.dt);
                shader.set("uColor", *color);
                mesh.draw(&shader);
            }
        }
    })
    .unwrap();
}
