use std::rc::Rc;

use glow::HasContext;

use crate::{
    geometry::{MESH_LAYOUT, MeshBuilder, Shape},
    shader::Shader,
    transform::Transform,
};

const F32: i32 = size_of::<f32>() as i32;

pub struct Mesh {
    pub transform: Transform,
    pub wireframe: bool,
    gl: Rc<glow::Context>,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    floats_per_vertex: i32,
    vertex_count: i32,
    shape: Option<Shape>,
    flat: bool,
}

impl Mesh {
    pub(crate) fn new(gl: Rc<glow::Context>, vertices: &[f32], layout: &[i32]) -> Self {
        let floats_per_vertex: i32 = layout.iter().sum();
        assert!(floats_per_vertex > 0, "layout vazio");

        let (vao, vbo) = unsafe {
            let vao = gl.create_vertex_array().expect("falha ao criar VAO");
            let vbo = gl.create_buffer().expect("falha ao criar VBO");
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            let stride = floats_per_vertex * F32;
            let mut offset = 0;
            for (i, &size) in layout.iter().enumerate() {
                gl.enable_vertex_attrib_array(i as u32);
                gl.vertex_attrib_pointer_f32(i as u32, size, glow::FLOAT, false, stride, offset);
                offset += size * F32;
            }

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            (vao, vbo)
        };

        let mut mesh = Self {
            transform: Transform::default(),
            wireframe: false,
            gl,
            vao,
            vbo,
            floats_per_vertex,
            vertex_count: 0,
            shape: None,
            flat: false,
        };
        mesh.upload(vertices);
        mesh
    }

    pub(crate) fn from_shape(gl: Rc<glow::Context>, shape: Shape) -> Self {
        let mut mesh = Self::new(gl, shape.build().data(), &MESH_LAYOUT);
        mesh.shape = Some(shape);
        mesh
    }

    fn upload(&mut self, vertices: &[f32]) {
        assert!(
            vertices.len() % self.floats_per_vertex as usize == 0,
            "vertices nao bate com o layout do mesh"
        );
        unsafe {
            let bytes = core::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                core::mem::size_of_val(vertices),
            );
            self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            self.gl
                .buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::DYNAMIC_DRAW);
            self.gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
        self.vertex_count = vertices.len() as i32 / self.floats_per_vertex;
    }

    fn rebuild(&mut self) {
        if let Some(shape) = self.shape {
            let mut builder = shape.build();
            if self.flat {
                builder.flatten();
            }
            self.upload(builder.data());
        }
    }

    pub fn shape(&self) -> Option<Shape> {
        self.shape
    }

    pub fn set_shape(&mut self, shape: Shape) {
        if self.shape != Some(shape) {
            self.shape = Some(shape);
            self.rebuild();
        }
    }

    pub fn segments(&self) -> Option<u32> {
        self.shape.and_then(|s| s.segments())
    }

    pub fn set_segments(&mut self, n: u32) {
        if let Some(shape) = self.shape {
            self.set_shape(shape.with_segments(n));
        }
    }

    pub fn flat_shading(&self) -> bool {
        self.flat
    }

    pub fn set_flat_shading(&mut self, flat: bool) {
        if self.flat != flat {
            self.flat = flat;
            self.rebuild();
        }
    }

    pub fn update(&mut self, builder: &MeshBuilder) {
        assert!(
            self.floats_per_vertex == MESH_LAYOUT.iter().sum::<i32>(),
            "update so serve para meshes posicao+normal"
        );
        self.shape = None;
        self.upload(builder.data());
    }

    pub fn vertex_count(&self) -> i32 {
        self.vertex_count
    }

    pub fn triangle_count(&self) -> i32 {
        self.vertex_count / 3
    }

    pub fn draw(&self, shader: &Shader) {
        shader.set("uModel", self.transform.matrix());
        shader.set("uNormalMatrix", self.transform.normal_matrix());
        unsafe {
            if self.wireframe {
                self.gl.polygon_mode(glow::FRONT_AND_BACK, glow::LINE);
            }
            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.draw_arrays(glow::TRIANGLES, 0, self.vertex_count);
            self.gl.bind_vertex_array(None);
            if self.wireframe {
                self.gl.polygon_mode(glow::FRONT_AND_BACK, glow::FILL);
            }
        }
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_vertex_array(self.vao);
            self.gl.delete_buffer(self.vbo);
        }
    }
}
