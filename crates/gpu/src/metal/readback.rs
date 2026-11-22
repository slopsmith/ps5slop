//! Reading textures back to the CPU, for screenshots and dumps.

use super::*;

/// `texture`'s pixels as RGBA8, read back for screenshots. Other 2D formats
/// are converted on the GPU, clamped to [0, 1] (integers scaled from 16 bits).
pub fn read_rgba(texture: &SharedTexture) -> Option<(u32, u32, Vec<u8>)> {
    let t = &texture.0;
    let (w, h) = (t.width(), t.height());
    let format = t.pixelFormat();
    let bgra = matches!(
        format,
        MTLPixelFormat::BGRA8Unorm | MTLPixelFormat::BGRA8Unorm_sRGB
    );
    if !bgra
        && !matches!(
            format,
            MTLPixelFormat::RGBA8Unorm | MTLPixelFormat::RGBA8Unorm_sRGB
        )
    {
        return read_converted(t, false);
    }
    let device = t.device();
    let queue = device.newCommandQueue()?;
    let buffer =
        device.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModeShared)?;
    let cb = queue.commandBuffer()?;
    let blit = cb.blitCommandEncoder()?;
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            t,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width: w, height: h, depth: 1 },
            &buffer,
            0,
            w * 4,
            w * h * 4,
        );
    }
    blit.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    let mut px =
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u8, w * h * 4) }
            .to_vec();
    if bgra {
        for p in px.as_chunks_mut::<4>().0 {
            p.swap(0, 2);
        }
    }
    Some((w as u32, h as u32, px))
}

/// The texels of a 2D texture as stored, row by row with no padding.
pub(super) fn read_raw(t: &Texture) -> Option<Vec<u8>> {
    let bpp = texel_bytes(t.pixelFormat())?;
    let (w, h) = (t.width(), t.height());
    let device = t.device();
    let queue = device.newCommandQueue()?;
    let buffer =
        device.newBufferWithLength_options(w * h * bpp, MTLResourceOptions::StorageModeShared)?;
    let cb = queue.commandBuffer()?;
    let blit = cb.blitCommandEncoder()?;
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            t, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: w, height: h, depth: 1 }, &buffer, 0, w * bpp, w * h * bpp,
        );
    }
    blit.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    Some(
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u8, w * h * bpp) }
            .to_vec(),
    )
}

/// A depth texture's pixels as stretched grayscale RGBA8.
pub fn read_depth_rgba(texture: &SharedTexture) -> Option<(u32, u32, Vec<u8>)> {
    read_depth(&texture.0)
}

/// A flipped frame's pixels as RGBA8, like [`read_rgba`], encoded as sRGB
/// first when the frame holds linear light.
pub fn read_frame(texture: &SharedTexture, linear: bool) -> Option<(u32, u32, Vec<u8>)> {
    if linear {
        read_converted(&texture.0, true)
    } else {
        read_rgba(texture)
    }
}

fn read_converted(t: &Texture, srgb: bool) -> Option<(u32, u32, Vec<u8>)> {
    if t.textureType() != MTLTextureType::Type2D || is_compressed(t.pixelFormat()) {
        return None;
    }
    let (w, h) = (t.width(), t.height());
    let (elem, scale) = match number(t.pixelFormat()) {
        Number::Float => ("float", "1.0"),
        Number::Uint => ("uint", "1.0 / 65535.0"),
        Number::Sint => ("int", "1.0 / 32767.0"),
    };
    let encode = if srgb {
        "v.rgb = select(1.055 * pow(v.rgb, 1.0 / 2.4) - 0.055, v.rgb * 12.92, v.rgb <= 0.0031308);"
    } else {
        ""
    };
    let msl = format!(
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void convert(texture2d<{elem}, access::read> src [[texture(0)]], device uchar4* dst [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {{\n    if (id.x >= src.get_width() || id.y >= src.get_height()) return;\n    float4 v = clamp(float4(src.read(id)) * {scale}, 0.0, 1.0);\n    {encode}\n    dst[id.y * src.get_width() + id.x] = uchar4(v * 255.0 + 0.5);\n}}\n"
    );
    let device = t.device();
    let lib = device
        .newLibraryWithSource_options_error(&NSString::from_str(&msl), None)
        .ok()?;
    let f = lib.newFunctionWithName(&NSString::from_str("convert"))?;
    let pipeline = device.newComputePipelineStateWithFunction_error(&f).ok()?;
    let queue = device.newCommandQueue()?;
    let buffer =
        device.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModeShared)?;
    let cb = queue.commandBuffer()?;
    let enc = cb.computeCommandEncoder()?;
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setTexture_atIndex(Some(t), 0);
        enc.setBuffer_offset_atIndex(Some(&buffer), 0, 0);
    }
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: w,
            height: h,
            depth: 1,
        },
        MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    let px =
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u8, w * h * 4) }
            .to_vec();
    Some((w as u32, h as u32, px))
}

/// An `R32Float` depth copy as grey RGBA8, stretched from its nearest to
/// its farthest depth below 1 (the cleared value) so near differences show.
pub(super) fn read_depth(t: &Texture) -> Option<(u32, u32, Vec<u8>)> {
    let (w, h) = (t.width(), t.height());
    let device = t.device();
    let queue = device.newCommandQueue()?;
    let buffer =
        device.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModeShared)?;
    let cb = queue.commandBuffer()?;
    let blit = cb.blitCommandEncoder()?;
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            t, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: w, height: h, depth: 1 }, &buffer, 0, w * 4, w * h * 4,
        );
    }
    blit.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    let z = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, w * h) };
    let (lo, hi) = z
        .iter()
        .filter(|v| **v < 1.0 && **v > 0.0)
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let scale = if hi > lo { 255.0 / (hi - lo) } else { 0.0 };
    let px = z
        .iter()
        .flat_map(|v| {
            let g = if *v >= 1.0 {
                255
            } else if *v <= 0.0 {
                0
            } else {
                ((v - lo) * scale).clamp(0.0, 254.0) as u8
            };
            [g, g, g, 255]
        })
        .collect();
    Some((w as u32, h as u32, px))
}
