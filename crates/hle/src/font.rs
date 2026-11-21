//! libSceFont and libSceFontFt: fonts for titles that draw their own text.
//! FreeType loads the fonts a title opens from memory, and the open-source
//! stand-in for system font sets (see `Vfs::font_set_file`), measures their
//! glyphs and renders them as 8-bit coverage into the title's surface. A
//! font with no face behind it (no stand-in installed) gets metrics from a
//! simple model and renders nothing.
//!
//! Libraries, renderers, fonts and glyphs are blocks of guest memory from
//! our heap, so titles get real pointers; their contents are ours and stay
//! zero. Prototypes, error codes and the layouts of the public structures
//! (glyph metrics, render surface, render output, horizontal layout) follow
//! shadPS4's PS4 notes; they are unverified for the PS5.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use freetype::face::LoadFlag;

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

const ERROR_INVALID_PARAMETER: u64 = 0x8046_0002;
const ERROR_INVALID_FONT_HANDLE: u64 = 0x8046_0005;
const ERROR_ALLOCATION_FAILED: u64 = 0x8046_0010;

/// Size of the guest blocks behind handles (a font handle is 256 bytes).
const HANDLE_SIZE: u64 = 0x100;
/// Pixel size of a font before the title sets one (our choice).
const DEFAULT_PIXELS: f32 = 16.0;

/// The pixel scale of a font: `w` across, `h` up.
#[derive(Clone, Copy)]
struct Scale {
    w: f32,
    h: f32,
}

#[derive(Clone, Copy)]
struct Font {
    scale: Scale,
    /// Set with `sceFontSetupRenderScalePixel`; rendering uses it over `scale`.
    render_scale: Option<Scale>,
    /// The FreeType face behind it, by its key in [`FreeType::faces`].
    face: Option<u64>,
}

type Face = freetype::Face<Arc<[u8]>>;

struct FreeType {
    library: freetype::Library,
    faces: HashMap<u64, Face>,
    next: u64,
}

// FreeType objects may move between threads as long as one thread at a
// time uses them; the mutex in `Fonts` sees to that.
unsafe impl Send for FreeType {}

#[derive(Default)]
pub struct Fonts {
    fonts: Mutex<HashMap<u64, Font>>,
    freetype: Mutex<Option<FreeType>>,
}

/// A glyph measured (and maybe rendered) at some scale, in pixels.
#[derive(Default)]
struct Glyph {
    width: f32,
    height: f32,
    bearing_x: f32,
    bearing_y: f32,
    advance: f32,
    vertical: [f32; 3],
    /// Coverage rows, width, height, and the top left corner relative to
    /// the pen.
    bitmap: Option<(Vec<u8>, usize, usize, i32, i32)>,
}

pub static FUNCTIONS: &[Function] = &[
    f("sceFontMemoryInit", ok),
    f("sceFontMemoryTerm", ok),
    ft("sceFontSelectLibraryFt", select),
    ft("sceFontSelectRendererFt", select),
    f("sceFontCreateLibraryWithEdition", create_object::<3>),
    f("sceFontCreateRendererWithEdition", create_object::<3>),
    f("sceFontDestroyLibrary", destroy_object),
    f("sceFontDestroyRenderer", destroy_object),
    f("sceFontSupportSystemFonts", ok),
    f("sceFontSupportExternalFonts", ok),
    f("sceFontAttachDeviceCacheBuffer", ok),
    f("sceFontOpenFontSet", open_font_set),
    f("sceFontOpenFontMemory", open_font_memory),
    f("sceFontOpenFontInstance", open_instance),
    f("sceFontCloseFont", close_font),
    f("sceFontBindRenderer", ok),
    f("sceFontUnbindRenderer", ok),
    f("sceFontSetScalePixel", set_scale_pixel),
    f("sceFontSetupRenderScalePixel", setup_render_scale_pixel),
    f("sceFontSetEffectSlant", ok),
    f("sceFontSetEffectWeight", ok),
    f("sceFontSetupRenderEffectSlant", ok),
    f("sceFontSetupRenderEffectWeight", ok),
    f("sceFontGetHorizontalLayout", horizontal_layout),
    f("sceFontGetVerticalLayout", vertical_layout),
    f("sceFontGetRenderCharGlyphMetrics", render_metrics),
    f("sceFontRenderSurfaceInit", surface_init),
    f("sceFontRenderSurfaceSetScissor", surface_set_scissor),
    f("sceFontRenderCharGlyphImageHorizontal", render_horizontal),
    f("sceFontGenerateCharGlyph", generate_glyph),
    f("sceFontGlyphDefineAttribute", ok),
    f("sceFontDeleteGlyph", delete_glyph),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceFont",
        name,
        func,
    }
}

const fn ft(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceFontFt",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

/// Float argument `n` (XMM`n`).
fn float_arg(env: &Env, n: usize) -> f32 {
    f32::from_bits(env.cpu.xmm[n][0] as u32)
}

fn new_block(env: &Env) -> Option<u64> {
    env.kernel.heap.alloc(env.mem, HANDLE_SIZE)
}

/// `sceFontSelectLibraryFt(value)` and `sceFontSelectRendererFt(value)`: a
/// selection for the create calls, which we do not look at.
fn select(env: &mut Env) -> Result<Outcome, CallError> {
    ret(new_block(env).unwrap_or(0))
}

/// The create calls that write a new object to the pointer in argument `N`.
fn create_object<const N: usize>(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(N) == 0 {
        return ret(ERROR_INVALID_PARAMETER);
    }
    let Some(object) = new_block(env) else {
        return ret(ERROR_ALLOCATION_FAILED);
    };
    env.mem.write_u64(env.arg(N), object)?;
    ret(0)
}

/// `sceFontDestroyLibrary(*library)` and `sceFontDestroyRenderer(*renderer)`.
fn destroy_object(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(0) == 0 {
        return ret(ERROR_INVALID_PARAMETER);
    }
    let object = env.mem.read_u64(env.arg(0))?;
    if object != 0 {
        env.kernel.heap.free(object);
    }
    env.mem.write_u64(env.arg(0), 0)?;
    ret(0)
}

fn add_font(env: &Env, font: Font, out: u64) -> Result<Outcome, CallError> {
    if out == 0 {
        return ret(ERROR_INVALID_PARAMETER);
    }
    let Some(handle) = new_block(env) else {
        return ret(ERROR_ALLOCATION_FAILED);
    };
    lock(&env.kernel.fonts.fonts).insert(handle, font);
    env.mem.write_u64(out, handle)?;
    ret(0)
}

fn new_font(face: Option<u64>) -> Font {
    let scale = Scale {
        w: DEFAULT_PIXELS,
        h: DEFAULT_PIXELS,
    };
    Font {
        scale,
        render_scale: None,
        face,
    }
}

/// Makes a FreeType face from font file bytes; `None` if FreeType rejects
/// them.
fn load_face(env: &Env, data: Vec<u8>) -> Option<u64> {
    let mut freetype = lock(&env.kernel.fonts.freetype);
    if freetype.is_none() {
        let library = freetype::Library::init().ok()?;
        *freetype = Some(FreeType {
            library,
            faces: HashMap::new(),
            next: 1,
        });
    }
    let ft = freetype.as_mut()?;
    let face = ft
        .library
        .new_memory_face2(Arc::<[u8]>::from(data), 0)
        .ok()?;
    let key = ft.next;
    ft.next += 1;
    ft.faces.insert(key, face);
    Some(key)
}

/// `sceFontOpenFontSet(library, set, mode, params, *font)`: every set is the
/// same stand-in font.
fn open_font_set(env: &mut Env) -> Result<Outcome, CallError> {
    let data = env
        .kernel
        .vfs
        .font_set_file()
        .and_then(|path| std::fs::read(path).ok());
    let face = data.and_then(|d| load_face(env, d));
    add_font(env, new_font(face), env.arg(4))
}

/// `sceFontOpenFontMemory(library, data, size, params, *font)`.
fn open_font_memory(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, size) = (env.arg(1), env.arg(2) as u32 as usize);
    if at == 0 || size == 0 {
        return ret(ERROR_INVALID_PARAMETER);
    }
    let mut data = vec![0; size];
    env.mem.read(at, &mut data)?;
    let face = load_face(env, data);
    add_font(env, new_font(face), env.arg(4))
}

/// `sceFontOpenFontInstance(font, template, *font)`: a copy of `font`.
fn open_instance(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(font) = font(env, env.arg(0)) else {
        return ret(ERROR_INVALID_FONT_HANDLE);
    };
    add_font(env, font, env.arg(2))
}

fn close_font(env: &mut Env) -> Result<Outcome, CallError> {
    let handle = env.arg(0);
    if lock(&env.kernel.fonts.fonts).remove(&handle).is_none() {
        return ret(ERROR_INVALID_FONT_HANDLE);
    }
    env.kernel.heap.free(handle);
    ret(0)
}

fn font(env: &Env, handle: u64) -> Option<Font> {
    lock(&env.kernel.fonts.fonts).get(&handle).copied()
}

fn change_font(env: &Env, change: impl FnOnce(&mut Font)) -> Result<Outcome, CallError> {
    match lock(&env.kernel.fonts.fonts).get_mut(&env.arg(0)) {
        Some(font) => {
            change(font);
            ret(0)
        }
        None => ret(ERROR_INVALID_FONT_HANDLE),
    }
}

/// `sceFontSetScalePixel(font, w, h)`.
fn set_scale_pixel(env: &mut Env) -> Result<Outcome, CallError> {
    let (w, h) = (float_arg(env, 0), float_arg(env, 1));
    change_font(env, |f| f.scale = Scale { w, h })
}

/// `sceFontSetupRenderScalePixel(font, w, h)`.
fn setup_render_scale_pixel(env: &mut Env) -> Result<Outcome, CallError> {
    let (w, h) = (float_arg(env, 0), float_arg(env, 1));
    change_font(env, |f| f.render_scale = Some(Scale { w, h }))
}

/// A glyph from our model of a proportional font, for fonts with no face.
fn glyph_model(code: u32, s: Scale) -> Glyph {
    let (_, line) = line_model(s);
    if char::from_u32(code).is_none_or(char::is_whitespace) {
        return Glyph {
            advance: 0.3 * s.w,
            vertical: [0.0, 0.0, line],
            ..Glyph::default()
        };
    }
    Glyph {
        width: 0.5 * s.w,
        height: 0.7 * s.h,
        bearing_x: 0.05 * s.w,
        bearing_y: 0.7 * s.h,
        advance: 0.6 * s.w,
        vertical: [-0.25 * s.w, 0.0, line],
        bitmap: None,
    }
}

/// Ascent and line advance at scale `s` in our model.
fn line_model(s: Scale) -> (f32, f32) {
    (0.8 * s.h, 1.25 * s.h)
}

/// 26.6 fixed point to pixels.
fn px(v: i64) -> f32 {
    v as f32 / 64.0
}

/// Runs `f` on `font`'s face sized to `s`, if it has one.
fn with_face<T>(env: &Env, font: Font, s: Scale, f: impl FnOnce(&Face) -> Option<T>) -> Option<T> {
    let freetype = lock(&env.kernel.fonts.freetype);
    let face = freetype.as_ref()?.faces.get(&font.face?)?;
    let (w, h) = ((s.w * 64.0) as isize, (s.h * 64.0) as isize);
    face.set_char_size(w.max(64), h.max(64), 72, 72).ok()?;
    f(face)
}

/// Glyph `code` of `font` at scale `s`, with its coverage if `render`.
fn glyph(env: &Env, font: Font, code: u32, s: Scale, render: bool) -> Glyph {
    let found = with_face(env, font, s, |face| {
        let flags = if render {
            LoadFlag::RENDER
        } else {
            LoadFlag::DEFAULT
        };
        face.load_char(code as usize, flags).ok()?;
        let slot = face.glyph();
        let m = slot.metrics();
        let bitmap = render.then(|| {
            let b = slot.bitmap();
            let (w, h) = (b.width().max(0) as usize, b.rows().max(0) as usize);
            let pitch = b.pitch().unsigned_abs() as usize;
            let rows = b.buffer().chunks(pitch.max(1)).take(h);
            let pixels = rows
                .flat_map(|r| r[..w.min(r.len())].iter().copied())
                .collect();
            (pixels, w, h, slot.bitmap_left(), -slot.bitmap_top())
        });
        Some(Glyph {
            width: px(m.width),
            height: px(m.height),
            bearing_x: px(m.horiBearingX),
            bearing_y: px(m.horiBearingY),
            advance: px(m.horiAdvance),
            vertical: [px(m.vertBearingX), px(m.vertBearingY), px(m.vertAdvance)],
            bitmap,
        })
    });
    found.unwrap_or_else(|| glyph_model(code, s))
}

/// Ascent and line advance of `font` at scale `s`.
fn line(env: &Env, font: Font, s: Scale) -> (f32, f32) {
    with_face(env, font, s, |face| {
        let m = face.size_metrics()?;
        Some((px(m.ascender), px(m.height)))
    })
    .unwrap_or_else(|| line_model(s))
}

fn write_floats(env: &Env, at: u64, values: &[f32]) -> Result<(), CallError> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    env.mem.write(at, &bytes)?;
    Ok(())
}

/// `sceFontGetHorizontalLayout(font, *layout)`: baseline offset, line
/// advance and decoration extent.
fn horizontal_layout(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(font) = font(env, env.arg(0)) else {
        return ret(ERROR_INVALID_FONT_HANDLE);
    };
    let (ascent, advance) = line(env, font, font.scale);
    write_floats(env, env.arg(1), &[ascent, advance, 0.0])?;
    ret(0)
}

/// `sceFontGetVerticalLayout(font, *layout)`: baseline offset X, column
/// advance and decoration span.
fn vertical_layout(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(font) = font(env, env.arg(0)) else {
        return ret(ERROR_INVALID_FONT_HANDLE);
    };
    write_floats(env, env.arg(1), &[0.5 * font.scale.w, font.scale.w, 0.0])?;
    ret(0)
}

/// The glyph metrics structure: width, height, then bearing X, bearing Y
/// and advance for horizontal and for vertical writing.
fn write_metrics(env: &Env, at: u64, g: &Glyph) -> Result<(), CallError> {
    let [vx, vy, va] = g.vertical;
    let values = [
        g.width,
        g.height,
        g.bearing_x,
        g.bearing_y,
        g.advance,
        vx,
        vy,
        va,
    ];
    write_floats(env, at, &values)
}

/// `sceFontGetRenderCharGlyphMetrics(font, code, *metrics)`.
fn render_metrics(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(font) = font(env, env.arg(0)) else {
        return ret(ERROR_INVALID_FONT_HANDLE);
    };
    let s = font.render_scale.unwrap_or(font.scale);
    let g = glyph(env, font, env.arg(1) as u32, s, false);
    write_metrics(env, env.arg(2), &g)?;
    ret(0)
}

/// Offsets in the render surface (`buffer`, `widthByte`, `pixelSizeByte`,
/// flags, `width`, `height`, scissor rectangle as x0, y0, x1, y1).
const SURFACE_WIDTH_BYTE: u64 = 8;
const SURFACE_PIXEL_SIZE: u64 = 12;
const SURFACE_WIDTH: u64 = 16;
const SURFACE_SCISSOR: u64 = 24;

/// `sceFontRenderSurfaceInit(surface, buffer, width_byte, pixel_size,
/// width, height)`: the scissor covers the whole surface.
fn surface_init(env: &mut Env) -> Result<Outcome, CallError> {
    let s = env.arg(0);
    let (width, height) = (env.arg(4) as u32, env.arg(5) as u32);
    env.mem.write(s, &[0; 0x80])?;
    env.mem.write_u64(s, env.arg(1))?;
    env.mem
        .write(s + SURFACE_WIDTH_BYTE, &(env.arg(2) as u32).to_le_bytes())?;
    env.mem.write(s + SURFACE_PIXEL_SIZE, &[env.arg(3) as u8])?;
    let mut size = width.to_le_bytes().to_vec();
    size.extend(height.to_le_bytes());
    env.mem.write(s + SURFACE_WIDTH, &size)?;
    write_scissor(env, s, [0, 0, width, height])?;
    ret(0)
}

fn write_scissor(env: &Env, s: u64, rect: [u32; 4]) -> Result<(), CallError> {
    let bytes: Vec<u8> = rect.iter().flat_map(|v| v.to_le_bytes()).collect();
    env.mem.write(s + SURFACE_SCISSOR, &bytes)?;
    Ok(())
}

/// `sceFontRenderSurfaceSetScissor(surface, x, y, w, h)`, kept inside the
/// surface.
fn surface_set_scissor(env: &mut Env) -> Result<Outcome, CallError> {
    let s = env.arg(0);
    let size = env.mem.read_u64(s + SURFACE_WIDTH)?;
    let (width, height) = (size as u32, (size >> 32) as u32);
    let (x, y) = (env.arg(1) as i32, env.arg(2) as i32);
    let (w, h) = (env.arg(3) as i32, env.arg(4) as i32);
    let clamp = |v: i32, max: u32| v.clamp(0, max as i32) as u32;
    let x0 = clamp(x, width);
    let y0 = clamp(y, height);
    let x1 = clamp(x.saturating_add(w), width);
    let y1 = clamp(y.saturating_add(h), height);
    write_scissor(env, s, [x0, y0, x1, y1])?;
    ret(0)
}

/// `sceFontRenderCharGlyphImageHorizontal(font, code, surface, x, y,
/// *metrics, *result)`, pen position as floats; coverage goes to every byte.
fn render_horizontal(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(font) = font(env, env.arg(0)) else {
        return ret(ERROR_INVALID_FONT_HANDLE);
    };
    let (code, surface) = (env.arg(1) as u32, env.arg(2));
    let (metrics, result) = (env.arg(3), env.arg(4));
    let (pen_x, pen_y) = (float_arg(env, 0), float_arg(env, 1));
    let s = font.render_scale.unwrap_or(font.scale);
    let g = glyph(env, font, code, s, true);
    if metrics != 0 {
        write_metrics(env, metrics, &g)?;
    }
    let mut head = [0u8; 40];
    env.mem.read(surface, &mut head)?;
    let word = |at: u64| u32::from_le_bytes(head[at as usize..at as usize + 4].try_into().unwrap());
    let buffer = u64::from_le_bytes(head[..8].try_into().unwrap());
    let width_byte = word(SURFACE_WIDTH_BYTE) as u64;
    let pixel_size = head[SURFACE_PIXEL_SIZE as usize].max(1) as u64;
    let scissor: [u32; 4] = std::array::from_fn(|i| word(SURFACE_SCISSOR + 4 * i as u64));
    let (left, top, w, h) = match &g.bitmap {
        Some((_, w, h, dx, dy)) => (
            pen_x.floor() as i64 + *dx as i64,
            pen_y.floor() as i64 + *dy as i64,
            *w as i64,
            *h as i64,
        ),
        None => (
            (pen_x + g.bearing_x).floor() as i64,
            (pen_y - g.bearing_y).floor() as i64,
            g.width.ceil() as i64,
            g.height.ceil() as i64,
        ),
    };
    let clip = |v: i64, lo: u32, hi: u32| v.clamp(lo as i64, hi as i64) as u32;
    let x0 = clip(left, scissor[0], scissor[2]);
    let y0 = clip(top, scissor[1], scissor[3]);
    let x1 = clip(left + w, scissor[0], scissor[2]);
    let y1 = clip(top + h, scissor[1], scissor[3]);
    if let Some((pixels, bw, _, _, _)) = &g.bitmap
        && buffer != 0
        && x1 > x0
    {
        let span = (x1 - x0) as u64 * pixel_size;
        let mut row = vec![0u8; span as usize];
        for y in y0..y1 {
            let at = buffer + y as u64 * width_byte + x0 as u64 * pixel_size;
            env.mem.read(at, &mut row)?;
            let src = (y as i64 - top) as usize * bw;
            for x in x0..x1 {
                let c = pixels[src + (x as i64 - left) as usize];
                let px = (x - x0) as usize * pixel_size as usize;
                for b in &mut row[px..px + pixel_size as usize] {
                    *b = (*b).max(c);
                }
            }
            env.mem.write(at, &row)?;
        }
    }
    if result == 0 {
        return ret(0);
    }
    let image = buffer + y0 as u64 * width_byte + x0 as u64 * pixel_size;
    let mut out = Vec::with_capacity(64);
    out.extend(0u64.to_le_bytes());
    out.extend(image.to_le_bytes());
    out.extend((width_byte as u32).to_le_bytes());
    out.extend([pixel_size as u8, 0, 0, 0]);
    for v in [x0, y0, x1 - x0, y1 - y0] {
        out.extend(v.to_le_bytes());
    }
    for v in [g.bearing_x, g.bearing_y, g.advance, width_byte as f32] {
        out.extend(v.to_le_bytes());
    }
    out.extend((x1 - x0).to_le_bytes());
    out.extend((y1 - y0).to_le_bytes());
    env.mem.write(result, &out)?;
    ret(0)
}

/// `sceFontGenerateCharGlyph(font, code, params, *glyph)`.
fn generate_glyph(env: &mut Env) -> Result<Outcome, CallError> {
    if font(env, env.arg(0)).is_none() {
        return ret(ERROR_INVALID_FONT_HANDLE);
    }
    create_object::<3>(env)
}

/// `sceFontDeleteGlyph(memory, *glyph)`.
fn delete_glyph(env: &mut Env) -> Result<Outcome, CallError> {
    let at = env.arg(1);
    if at == 0 {
        return ret(ERROR_INVALID_PARAMETER);
    }
    let glyph = env.mem.read_u64(at)?;
    if glyph != 0 {
        env.kernel.heap.free(glyph);
    }
    env.mem.write_u64(at, 0)?;
    ret(0)
}
