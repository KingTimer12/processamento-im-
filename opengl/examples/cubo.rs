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
    run(Config::new("Cubo", 800, 600), |gl| {
        let shader = gl.shader(VERT, FRAG);
        let mut cube = gl.cube();

        move |gl, frame| {
            if frame.key_down(KeyCode::Escape) {
                frame.exit();
            }

            let proj = perspective(45.0, frame.aspect(), 0.1, 50.0);
            let view = look_at(Vec3::new(3.0, 3.0, 5.0), Vec3::ZERO);
            cube.transform.spin(Vec3::Y, 60.0, frame.dt);
            cube.transform.spin(Vec3::X, 30.0, frame.dt);

            gl.clear(0.0, 0.0, 0.0);
            shader.set("uProj", proj);
            shader.set("uView", view);
            shader.set("uColor", Vec3::new(0.0, 0.4, 0.8));
            cube.draw(&shader);
        }
    })
    .unwrap();
}
