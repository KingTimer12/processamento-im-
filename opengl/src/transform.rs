use glam::{Mat3, Mat4, Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn rotate(&mut self, axis: Vec3, degrees: f32) -> &mut Self {
        self.rotation = (Quat::from_axis_angle(axis.normalize(), degrees.to_radians()) * self.rotation).normalize();
        self
    }

    pub fn rotate_x(&mut self, degrees: f32) -> &mut Self {
        self.rotate(Vec3::X, degrees)
    }

    pub fn rotate_y(&mut self, degrees: f32) -> &mut Self {
        self.rotate(Vec3::Y, degrees)
    }

    pub fn rotate_z(&mut self, degrees: f32) -> &mut Self {
        self.rotate(Vec3::Z, degrees)
    }

    pub fn spin(&mut self, axis: Vec3, degrees_per_sec: f32, dt: f32) -> &mut Self {
        self.rotate(axis, degrees_per_sec * dt)
    }

    pub fn set_rotation(&mut self, axis: Vec3, degrees: f32) -> &mut Self {
        self.rotation = Quat::from_axis_angle(axis.normalize(), degrees.to_radians());
        self
    }

    pub fn translate(&mut self, delta: Vec3) -> &mut Self {
        self.position += delta;
        self
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    pub fn normal_matrix(&self) -> Mat3 {
        Mat3::from_mat4(self.matrix().inverse().transpose())
    }
}
