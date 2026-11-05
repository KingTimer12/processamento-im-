use std::rc::Rc;

use glow::HasContext;

use crate::{
    geometry::{MESH_LAYOUT, MeshBuilder, Shape},
    mesh::Mesh,
    shader::Shader,
};

#[derive(Clone)]
pub struct Gfx {
    gl: Rc<glow::Context>,
}

impl Gfx {
    pub(crate) fn new(gl: glow::Context) -> Self {
        unsafe { gl.enable(glow::DEPTH_TEST) }
        Self { gl: Rc::new(gl) }
    }

    pub fn clear(&self, r: f32, g: f32, b: f32) {
        unsafe {
            self.gl.clear_color(r, g, b, 1.0);
            self.gl
                .clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
        }
    }

    pub fn shader(&self, vertex_src: &str, fragment_src: &str) -> Shader {
        Shader::new(self.gl.clone(), vertex_src, fragment_src)
    }

    pub fn mesh(&self, vertices: &[f32], layout: &[i32]) -> Mesh {
        Mesh::new(self.gl.clone(), vertices, layout)
    }

    pub fn build(&self, builder: &MeshBuilder) -> Mesh {
        self.mesh(builder.data(), &MESH_LAYOUT)
    }

    pub fn shape(&self, shape: Shape) -> Mesh {
        Mesh::from_shape(self.gl.clone(), shape)
    }

    pub fn cube(&self) -> Mesh {
        self.shape(Shape::Cube { size: 2.0 })
    }

    pub fn plane(&self, width: f32, depth: f32) -> Mesh {
        self.shape(Shape::Plane { width, depth })
    }

    pub fn regular_polygon(&self, sides: u32, radius: f32) -> Mesh {
        self.shape(Shape::RegularPolygon { sides, radius })
    }

    pub fn prism(&self, sides: u32, radius: f32, height: f32) -> Mesh {
        self.shape(Shape::Prism { sides, radius, height })
    }

    pub fn pyramid(&self, sides: u32, radius: f32, height: f32) -> Mesh {
        self.shape(Shape::Pyramid { sides, radius, height })
    }

    pub fn cylinder(&self, radius: f32, height: f32, segments: u32) -> Mesh {
        self.shape(Shape::Cylinder { radius, height, segments })
    }

    pub fn cone(&self, radius: f32, height: f32, segments: u32) -> Mesh {
        self.shape(Shape::Cone { radius, height, segments })
    }

    pub fn sphere(&self, radius: f32, segments: u32) -> Mesh {
        self.shape(Shape::Sphere { radius, segments })
    }

    pub fn depth_test(&self, on: bool) {
        unsafe {
            if on {
                self.gl.enable(glow::DEPTH_TEST)
            } else {
                self.gl.disable(glow::DEPTH_TEST)
            }
        }
    }

    pub fn viewport(&self, width: i32, height: i32) {
        unsafe { self.gl.viewport(0, 0, width, height) }
    }

    pub fn raw(&self) -> &glow::Context {
        &self.gl
    }
}
