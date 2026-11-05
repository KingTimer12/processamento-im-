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
    run(Config::new("Suavizar", 1100, 500), |gl| {
        let shader = gl.shader(VERT, FRAG);

        let mut wire = gl.sphere(1.0, 3);
        wire.wireframe = true;
        wire.transform.position = Vec3::new(-3.0, 0.0, 0.0);

        let mut flat = gl.sphere(1.0, 3);
        flat.set_flat_shading(true);

        let mut smooth = gl.sphere(1.0, 3);
        smooth.transform.position = Vec3::new(3.0, 0.0, 0.0);

        let mut meshes = [wire, flat, smooth];

        move |gl, frame| {
            if frame.key_down(KeyCode::Escape) {
                frame.exit();
            }

            let segments = 3 + (frame.time * 3.0) as u32 % 46;
            gl.clear(0.05, 0.05, 0.08);
            shader.set("uProj", perspective(45.0, frame.aspect(), 0.1, 100.0));
            shader.set("uView", look_at(Vec3::new(0.0, 1.5, 9.0), Vec3::ZERO));
            shader.set("uColor", Vec3::new(0.2, 0.6, 0.9));

            for mesh in meshes.iter_mut() {
                mesh.set_segments(segments);
                mesh.transform.spin(Vec3::Y, 30.0, frame.dt);
                mesh.draw(&shader);
            }
        }
    })
    .unwrap();
}
