use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use glow::HasContext;

use crate::uniform::Uniform;

pub struct Shader {
    gl: Rc<glow::Context>,
    pub(crate) program: glow::Program,
    locations: RefCell<HashMap<String, Option<glow::UniformLocation>>>,
}

impl Shader {
    pub(crate) fn new(gl: Rc<glow::Context>, vertex_src: &str, fragment_src: &str) -> Self {
        let program = unsafe {
            let program = gl.create_program().expect("falha ao criar programa");
            let stages = [
                (glow::VERTEX_SHADER, vertex_src, "vertex"),
                (glow::FRAGMENT_SHADER, fragment_src, "fragment"),
            ];

            let mut shaders = Vec::new();
            for (kind, src, name) in stages {
                let shader = gl.create_shader(kind).expect("falha ao criar shader");
                gl.shader_source(shader, src);
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    panic!("erro no {name} shader: {}", gl.get_shader_info_log(shader));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }

            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                panic!("erro no link: {}", gl.get_program_info_log(program));
            }

            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            program
        };

        Self {
            gl,
            program,
            locations: RefCell::new(HashMap::new()),
        }
    }

    pub fn bind(&self) {
        unsafe { self.gl.use_program(Some(self.program)) }
    }

    pub fn set(&self, name: &str, value: impl Into<Uniform>) {
        self.bind();
        let mut locations = self.locations.borrow_mut();
        let loc = locations
            .entry(name.to_owned())
            .or_insert_with(|| unsafe { self.gl.get_uniform_location(self.program, name) });
        unsafe { value.into().upload(&self.gl, loc.as_ref()) }
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { self.gl.delete_program(self.program) }
    }
}
