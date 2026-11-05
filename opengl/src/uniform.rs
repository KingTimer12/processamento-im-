use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use glow::HasContext;

pub enum Uniform {
    F32(f32),
    I32(i32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Mat3(Mat3),
    Mat4(Mat4),
}

macro_rules! impl_from {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(impl From<$ty> for Uniform {
            fn from(v: $ty) -> Self {
                Uniform::$variant(v)
            }
        })*
    };
}

impl_from! {
    f32 => F32,
    i32 => I32,
    Vec2 => Vec2,
    Vec3 => Vec3,
    Vec4 => Vec4,
    Mat3 => Mat3,
    Mat4 => Mat4,
}

impl From<[f32; 3]> for Uniform {
    fn from(v: [f32; 3]) -> Self {
        Uniform::Vec3(Vec3::from(v))
    }
}

impl Uniform {
    pub(crate) unsafe fn upload(&self, gl: &glow::Context, loc: Option<&glow::UniformLocation>) {
        unsafe {
            match self {
                Uniform::F32(v) => gl.uniform_1_f32(loc, *v),
                Uniform::I32(v) => gl.uniform_1_i32(loc, *v),
                Uniform::Vec2(v) => gl.uniform_2_f32(loc, v.x, v.y),
                Uniform::Vec3(v) => gl.uniform_3_f32(loc, v.x, v.y, v.z),
                Uniform::Vec4(v) => gl.uniform_4_f32(loc, v.x, v.y, v.z, v.w),
                Uniform::Mat3(m) => gl.uniform_matrix_3_f32_slice(loc, false, &m.to_cols_array()),
                Uniform::Mat4(m) => gl.uniform_matrix_4_f32_slice(loc, false, &m.to_cols_array()),
            }
        }
    }
}
