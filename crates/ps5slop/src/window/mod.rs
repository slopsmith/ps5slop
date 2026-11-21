//! The window: winit for the window and keyboard, Metal for drawing. It
//! runs on the main thread (AppKit requires it); the guest runs on its own.

mod feedback;
mod gamepad;
mod layout;
mod present;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hle::{Kernel, PadInput};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::{Booted, Error};
use layout::window_size_for;
use present::Presenter;

/// The PS5's usual output size, for the first window.
const FIRST_SIZE: (u32, u32) = (1920, 1080);
/// How often controllers are polled.
const PAD_POLL: Duration = Duration::from_millis(8);
/// How often the title's instruction rates are refreshed.
const TITLE_INTERVAL: Duration = Duration::from_millis(500);

enum Wake {
    Frame,
    GuestDone,
}

/// What the main thread waits for before it opens a window.
enum Start {
    VideoOpened,
    GuestDone,
}

/// Runs `booted` on a new thread. Once the title opens its video output, a
/// window shows its frames until the guest exits or the window closes (which
/// ends it). Returns the guest's exit status.
pub fn run(booted: Booted, label: String) -> Result<i32, Error> {
    let kernel = booted.kernel.clone();
    let (start, started) = std::sync::mpsc::channel();
    let opened = Mutex::new(start.clone());
    kernel.set_open_listener(Box::new(move || {
        let _ = opened
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send(Start::VideoOpened);
    }));
    let proxy: Arc<Mutex<Option<EventLoopProxy<Wake>>>> = Arc::default();
    let result = Arc::new(Mutex::new(None));
    let (slot, done_proxy) = (result.clone(), proxy.clone());
    let guest = std::thread::spawn(move || {
        *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(booted.run());
        let _ = start.send(Start::GuestDone);
        if let Some(p) = done_proxy
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            let _ = p.send_event(Wake::GuestDone);
        }
    });
    let take = |guest: std::thread::JoinHandle<()>| {
        let _ = guest.join();
        result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .unwrap_or(Ok(0))
    };
    if !matches!(started.recv(), Ok(Start::VideoOpened)) {
        return take(guest);
    }
    let event_loop = match EventLoop::<Wake>::with_user_event().build() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ps5slop: cannot open a window ({e}), running without one");
            return take(guest);
        }
    };
    gamepad::init();
    let frames = event_loop.create_proxy();
    kernel.set_flip_listener(Box::new(move || {
        let _ = frames.send_event(Wake::Frame);
    }));
    *proxy.lock().unwrap_or_else(|e| e.into_inner()) = Some(event_loop.create_proxy());
    if guest.is_finished() {
        let _ = event_loop.create_proxy().send_event(Wake::GuestDone);
    }
    let mut app = App {
        kernel: kernel.clone(),
        window: None,
        presenter: None,
        flips: 0,
        keys: PadInput::default(),
        sent: PadInput::default(),
        feedback: feedback::Feedback::default(),
        rate_at: (Instant::now(), kernel.instructions()),
        label,
        size: None,
        flips_then: 0,
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("ps5slop: window: {e}");
    }
    if !kernel.has_exited() {
        kernel.exit(Ok(0));
    }
    take(guest)
}

struct App {
    kernel: Arc<Kernel>,
    window: Option<Window>,
    presenter: Option<Presenter>,
    flips: u64,
    /// What the keyboard holds, and what the guest was last given.
    keys: PadInput,
    sent: PadInput,
    feedback: feedback::Feedback,
    /// When the title was last refreshed, and the instruction counts then.
    rate_at: (Instant, (u64, u64)),
    /// The game's name, version and title id.
    label: String,
    /// The size of the last frame shown.
    size: Option<(u32, u32)>,
    /// `flips` when the title was last refreshed.
    flips_then: u64,
}

/// The pad button a key stands for, as `ORBIS_PAD_BUTTON_*` bits.
fn button(code: KeyCode) -> Option<u32> {
    Some(match code {
        KeyCode::ArrowUp => 0x0010,
        KeyCode::ArrowRight => 0x0020,
        KeyCode::ArrowDown => 0x0040,
        KeyCode::ArrowLeft => 0x0080,
        KeyCode::Digit1 => 0x0100,
        KeyCode::Digit3 => 0x0200,
        KeyCode::KeyQ => 0x0400,
        KeyCode::KeyE => 0x0800,
        KeyCode::KeyV => 0x1000,
        KeyCode::KeyC => 0x2000,
        KeyCode::KeyX => 0x4000,
        KeyCode::KeyZ => 0x8000,
        KeyCode::Enter => 0x0008,
        KeyCode::KeyT => 0x10_0000,
        _ => return None,
    })
}

const L2: u32 = 0x0100;
const R2: u32 = 0x0200;

impl App {
    fn key(&mut self, ev: KeyEvent) {
        let PhysicalKey::Code(code) = ev.physical_key else {
            return;
        };
        let Some(bit) = button(code) else { return };
        if ev.state == ElementState::Pressed {
            self.keys.buttons |= bit;
        } else {
            self.keys.buttons &= !bit;
        }
        self.keys.l2 = if self.keys.buttons & L2 != 0 { 255 } else { 0 };
        self.keys.r2 = if self.keys.buttons & R2 != 0 { 255 } else { 0 };
        self.update_pad();
    }

    /// Shows the game, its frame rate and size, the guest's instruction rates
    /// and the controller in the title, every [`TITLE_INTERVAL`].
    fn retitle(&mut self) {
        let (at, (interp, jit)) = self.rate_at;
        let elapsed = at.elapsed();
        if elapsed < TITLE_INTERVAL {
            return;
        }
        let now = self.kernel.instructions();
        let secs = elapsed.as_secs_f64();
        let mips = |a: u64, b: u64| (b - a) as f64 / secs / 1e6;
        let fps = (self.flips - self.flips_then) as f64 / secs;
        let mut parts = vec![self.label.clone(), format!("{fps:.0} fps")];
        if let Some((w, h)) = self.size {
            parts.push(format!("{w}x{h}"));
        }
        parts.push(format!(
            "JIT {:.0} MIPS, interpreter {:.1}",
            mips(jit, now.1),
            mips(interp, now.0)
        ));
        parts.push(gamepad::current().map_or("no controller".into(), |c| feedback::kind(&c)));
        if let Some(w) = &self.window {
            w.set_title(&parts.join(" | "));
        }
        self.rate_at = (Instant::now(), now);
        self.flips_then = self.flips;
    }

    /// Gives the guest the keyboard and controller state, if it changed, and
    /// the controller what the guest asked of it.
    fn update_pad(&mut self) {
        let controller = gamepad::current();
        self.feedback
            .apply(controller.as_ref(), self.kernel.pad_output());
        let input = gamepad::merge(controller.and_then(|c| gamepad::read(&c)), self.keys);
        if input != self.sent {
            self.sent = input;
            self.kernel.set_pad_input(input);
        }
    }

    fn take_frame(&mut self) {
        let Some(frame) = self.kernel.frame() else {
            return;
        };
        if frame.flips == self.flips {
            return;
        }
        self.flips = frame.flips;
        self.size = Some((frame.width, frame.height));
        if let (Some(p), Some(w)) = (&mut self.presenter, &self.window) {
            match &frame.texture {
                Some(t) => p.show(&t.0, frame.linear),
                None => p.upload(&frame.rgba, frame.width, frame.height),
            }
            w.request_redraw();
        }
    }
}

impl ApplicationHandler<Wake> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let screen = event_loop.primary_monitor().map(|m| {
            let size = m.size().to_logical::<f64>(m.scale_factor());
            (size.width, size.height)
        });
        let (w, h) = window_size_for(FIRST_SIZE, screen);
        let attrs = Window::default_attributes()
            .with_title(&self.label)
            .with_inner_size(LogicalSize::new(w, h))
            .with_resizable(true);
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("ps5slop: cannot open a window: {e}");
                event_loop.exit();
                return;
            }
        };
        match Presenter::new(&window) {
            Ok(p) => self.presenter = Some(p),
            Err(e) => {
                eprintln!("ps5slop: cannot draw into the window: {e}");
                event_loop.exit();
                return;
            }
        }
        self.window = Some(window);
        self.take_frame();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.update_pad();
        self.retitle();
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + PAD_POLL));
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, wake: Wake) {
        match wake {
            Wake::Frame => self.take_frame(),
            Wake::GuestDone => event_loop.exit(),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let (Some(p), Some(w)) = (&self.presenter, &self.window) {
                    p.resize(size.into(), w.scale_factor());
                    w.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let (Some(p), Some(w)) = (&self.presenter, &self.window) {
                    p.resize(w.inner_size().into(), scale_factor);
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(p) = &mut self.presenter {
                    p.draw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => self.key(event),
            WindowEvent::Focused(false) => {
                self.keys.buttons = 0;
                self.keys.l2 = 0;
                self.keys.r2 = 0;
                self.update_pad();
            }
            _ => {}
        }
    }
}
