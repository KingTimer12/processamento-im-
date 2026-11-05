use std::cell::Cell;
use std::collections::HashSet;
use std::error::Error;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::time::Instant;

use glutin::config::{Config as GlConfig, ConfigTemplateBuilder, GetGlConfig, GlConfig as _};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentContext, PossiblyCurrentContext, Version,
};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::prelude::*;
use glutin::surface::{Surface, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::gfx::Gfx;

#[derive(Clone)]
pub struct Config {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Config {
    pub fn new(title: &str, width: u32, height: u32) -> Self {
        Self {
            title: title.to_owned(),
            width,
            height,
        }
    }

    fn window_attributes(&self) -> WindowAttributes {
        Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height))
    }
}

pub struct Frame {
    pub time: f32,
    pub dt: f32,
    pub width: u32,
    pub height: u32,
    keys: HashSet<KeyCode>,
    exit: Cell<bool>,
}

impl Frame {
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }

    pub fn key_down(&self, key: KeyCode) -> bool {
        self.keys.contains(&key)
    }

    pub fn exit(&self) {
        self.exit.set(true);
    }
}

pub fn run<F>(config: Config, setup: impl FnOnce(&Gfx) -> F + 'static) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&Gfx, &Frame) + 'static,
{
    let event_loop = EventLoop::new()?;
    let display_builder = DisplayBuilder::new().with_window_attributes(Some(config.window_attributes()));

    let mut app = App {
        config,
        template: ConfigTemplateBuilder::new(),
        display_builder: Some(display_builder),
        setup: Some(Box::new(move |gfx| Box::new(setup(gfx)))),
        frame_fn: None,
        gfx: None,
        state: None,
        gl_context: None,
        keys: HashSet::new(),
        start: Instant::now(),
        last: Instant::now(),
        exit_state: Ok(()),
    };
    event_loop.run_app(&mut app)?;
    app.exit_state
}

type FrameFn = Box<dyn FnMut(&Gfx, &Frame)>;

struct AppState {
    gl_surface: Surface<WindowSurface>,
    window: Window,
}

struct App {
    config: Config,
    template: ConfigTemplateBuilder,
    display_builder: Option<DisplayBuilder>,
    setup: Option<Box<dyn FnOnce(&Gfx) -> FrameFn>>,
    frame_fn: Option<FrameFn>,
    gfx: Option<Gfx>,
    state: Option<AppState>,
    gl_context: Option<PossiblyCurrentContext>,
    keys: HashSet<KeyCode>,
    start: Instant,
    last: Instant,
    exit_state: Result<(), Box<dyn Error>>,
}

fn pick_config(configs: Box<dyn Iterator<Item = GlConfig> + '_>) -> GlConfig {
    configs
        .reduce(|a, b| if b.num_samples() > a.num_samples() { b } else { a })
        .unwrap()
}

fn create_context(window: &Window, gl_config: &GlConfig) -> NotCurrentContext {
    let raw = window.window_handle().ok().map(|h| h.as_raw());
    let display = gl_config.display();

    let attempts = [
        ContextAttributesBuilder::new().build(raw),
        ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(None))
            .build(raw),
        ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
            .build(raw),
    ];

    attempts
        .iter()
        .find_map(|attrs| unsafe { display.create_context(gl_config, attrs).ok() })
        .expect("falha ao criar contexto GL")
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let (window, gl_config) = match self.display_builder.take() {
            Some(builder) => {
                let (window, gl_config) = match builder.build(event_loop, self.template.clone(), pick_config) {
                    Ok((window, gl_config)) => (window.unwrap(), gl_config),
                    Err(err) => {
                        self.exit_state = Err(err);
                        event_loop.exit();
                        return;
                    }
                };
                self.gl_context = Some(create_context(&window, &gl_config).treat_as_possibly_current());
                (window, gl_config)
            }
            None => {
                let gl_config = self.gl_context.as_ref().unwrap().config();
                match glutin_winit::finalize_window(event_loop, self.config.window_attributes(), &gl_config) {
                    Ok(window) => (window, gl_config),
                    Err(err) => {
                        self.exit_state = Err(err.into());
                        event_loop.exit();
                        return;
                    }
                }
            }
        };

        let attrs = window
            .build_surface_attributes(Default::default())
            .expect("falha ao montar surface attributes");
        let gl_surface = unsafe {
            gl_config
                .display()
                .create_window_surface(&gl_config, &attrs)
                .unwrap()
        };
        self.gl_context
            .as_ref()
            .unwrap()
            .make_current(&gl_surface)
            .unwrap();

        if self.gfx.is_none() {
            let display = gl_config.display();
            let gl = unsafe {
                glow::Context::from_loader_function(|s| {
                    let s = CString::new(s).unwrap();
                    display.get_proc_address(&s) as *const _
                })
            };
            let gfx = Gfx::new(gl);
            if let Some(setup) = self.setup.take() {
                self.frame_fn = Some(setup(&gfx));
            }
            self.gfx = Some(gfx);
            self.start = Instant::now();
            self.last = self.start;
        }

        self.state = Some(AppState { gl_surface, window });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width != 0 && size.height != 0 => {
                if let (Some(state), Some(ctx)) = (&self.state, &self.gl_context) {
                    state.gl_surface.resize(
                        ctx,
                        NonZeroU32::new(size.width).unwrap(),
                        NonZeroU32::new(size.height).unwrap(),
                    );
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => self.keys.insert(code),
                        ElementState::Released => self.keys.remove(&code),
                    };
                }
            }
            WindowEvent::RedrawRequested => {
                let (Some(state), Some(ctx), Some(gfx), Some(frame_fn)) = (
                    &self.state,
                    &self.gl_context,
                    &self.gfx,
                    &mut self.frame_fn,
                ) else {
                    return;
                };

                let size = state.window.inner_size();
                if size.width == 0 || size.height == 0 {
                    return;
                }
                gfx.viewport(size.width as i32, size.height as i32);

                let now = Instant::now();
                let frame = Frame {
                    time: (now - self.start).as_secs_f32(),
                    dt: (now - self.last).as_secs_f32(),
                    width: size.width,
                    height: size.height,
                    keys: self.keys.clone(),
                    exit: Cell::new(false),
                };
                self.last = now;

                frame_fn(gfx, &frame);
                state.gl_surface.swap_buffers(ctx).unwrap();

                if frame.exit.get() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
