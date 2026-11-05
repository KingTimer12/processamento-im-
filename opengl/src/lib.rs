mod app;
mod geometry;
mod gfx;
mod mesh;
mod shader;
mod transform;
mod uniform;

pub use app::{Config, Frame, run};
pub use geometry::{MeshBuilder, Shape};
pub use gfx::Gfx;
pub use mesh::Mesh;
pub use shader::Shader;
pub use transform::Transform;
pub use uniform::Uniform;

pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
pub use winit::keyboard::KeyCode;

pub mod prelude {
    pub use crate::{
        Config, Frame, Gfx, KeyCode, Mat3, Mat4, Mesh, MeshBuilder, Quat, Shader, Shape, Transform, Vec2, Vec3, Vec4, look_at,
        perspective, run,
    };
}

pub fn perspective(fov_y_degrees: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    glam::camera::rh::proj::opengl::perspective(fov_y_degrees.to_radians(), aspect, near, far)
}

pub fn look_at(eye: Vec3, target: Vec3) -> Mat4 {
    glam::camera::rh::view::look_at_mat4(eye, target, Vec3::Y)
}
