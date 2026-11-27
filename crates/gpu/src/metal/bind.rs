//! Binding a program's resources to an encoder.

use super::*;

/// How much of a table the user data points to is searched for V#s.
pub(super) const TABLE_DWORDS: usize = 64;

pub(super) struct Bound {
    memory: Option<Memory>,
    pub(super) gds: Option<Retained<Buffer>>,
    /// Dword 0 of each sampler slot's S#, by slot.
    sampler_words: Vec<(usize, [u32; 4])>,
    /// Dwords 1 and 3 of each texture slot's T#, by slot.
    texture_words: Vec<(usize, [u32; 2])>,
    pub(super) textures: Vec<(usize, Retained<Texture>)>,
    pub(super) samplers: Vec<(usize, Retained<ProtocolObject<dyn MTLSamplerState>>)>,
    /// Constant buffers: binding index, buffer and offset.
    constants: Vec<(usize, Retained<Buffer>, usize)>,
    /// Tables of texture handles (`binding::HEAPS`): binding index, buffer
    /// and offset.
    heaps: Vec<(usize, Retained<Buffer>, usize)>,
    /// Textures programs reach only through `heaps`.
    resident: Vec<Retained<Texture>>,
}

/// Entries of a table of T#s a program indexes at run time that are bound;
/// an index past them reads nothing.
const MAX_HEAP_ENTRIES: usize = 1 << 14;

/// Where a program's resources are bound: one stage of an encoder.
#[derive(Clone, Copy)]
pub(super) enum Slot<'a> {
    Vertex(&'a RenderEncoder),
    Mesh(&'a RenderEncoder),
    Fragment(&'a RenderEncoder),
    Compute(&'a ProtocolObject<dyn MTLComputeCommandEncoder>),
}

impl Slot<'_> {
    pub(super) fn bytes(self, data: *const std::ffi::c_void, len: usize, index: usize) {
        let data = NonNull::new(data as *mut std::ffi::c_void).expect("bound data is not null");
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexBytes_length_atIndex(data, len, index),
                Slot::Mesh(e) => e.setMeshBytes_length_atIndex(data, len, index),
                Slot::Fragment(e) => e.setFragmentBytes_length_atIndex(data, len, index),
                Slot::Compute(e) => e.setBytes_length_atIndex(data, len, index),
            }
        }
    }

    fn texture(self, t: &Texture, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexTexture_atIndex(Some(t), index),
                Slot::Mesh(e) => e.setMeshTexture_atIndex(Some(t), index),
                Slot::Fragment(e) => e.setFragmentTexture_atIndex(Some(t), index),
                Slot::Compute(e) => e.setTexture_atIndex(Some(t), index),
            }
        }
    }

    fn sampler(self, s: &ProtocolObject<dyn MTLSamplerState>, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexSamplerState_atIndex(Some(s), index),
                Slot::Mesh(e) => e.setMeshSamplerState_atIndex(Some(s), index),
                Slot::Fragment(e) => e.setFragmentSamplerState_atIndex(Some(s), index),
                Slot::Compute(e) => e.setSamplerState_atIndex(Some(s), index),
            }
        }
    }

    /// Makes `b` resident for programs that reach it only through GPU addresses.
    fn reach(self, b: &Buffer) {
        self.make_resident(
            ProtocolObject::from_ref(b),
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
    }

    fn make_resident(self, r: &ProtocolObject<dyn MTLResource>, usage: MTLResourceUsage) {
        match self {
            Slot::Vertex(e) => e.useResource_usage_stages(r, usage, MTLRenderStages::Vertex),
            Slot::Mesh(e) => e.useResource_usage_stages(r, usage, MTLRenderStages::Mesh),
            Slot::Fragment(e) => e.useResource_usage_stages(r, usage, MTLRenderStages::Fragment),
            Slot::Compute(e) => e.useResource_usage(r, usage),
        }
    }

    pub(super) fn buffer(self, b: &Buffer, offset: usize, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexBuffer_offset_atIndex(Some(b), offset, index),
                Slot::Mesh(e) => e.setMeshBuffer_offset_atIndex(Some(b), offset, index),
                Slot::Fragment(e) => e.setFragmentBuffer_offset_atIndex(Some(b), offset, index),
                Slot::Compute(e) => e.setBuffer_offset_atIndex(Some(b), offset, index),
            }
        }
    }
}

impl Metal {
    /// Descriptor dwords from user data or memory.
    pub(super) fn descriptor(
        &self,
        host: &dyn Host,
        user: &[u32],
        source: &Source,
        n: usize,
    ) -> Option<Vec<u32>> {
        match source {
            Source::UserSgpr(s) => user.get(*s as usize..*s as usize + n).map(<[u32]>::to_vec),
            Source::Memory { pointer, offset } => {
                let p = self.descriptor(host, user, pointer, 2)?;
                let ptr = p[0] as u64 | (p[1] as u64) << 32;
                let mut b = vec![0u8; n * 4];
                host.read(ptr.wrapping_add(*offset as u64), &mut b)
                    .then(|| {
                        b.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|c| u32::from_le_bytes(*c))
                            .collect()
                    })
            }
            Source::Buffer {
                buffer,
                offset,
                index,
            } => {
                let v = self.descriptor(host, user, buffer, 4)?;
                let base = v[0] as u64 | (v[1] as u64 & 0xFFFF) << 32;
                let indexed = match *index {
                    Some((ref of, shift)) => {
                        self.descriptor(host, user, of, 1)?[0].checked_shl(shift as u32)?
                    }
                    None => 0,
                };
                let mut b = vec![0u8; n * 4];
                host.read(
                    base.wrapping_add(*offset as u64)
                        .wrapping_add(indexed as u64),
                    &mut b,
                )
                .then(|| {
                    b.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes(*c))
                        .collect()
                })
            }
            Source::Scaled { .. } | Source::Indexed { .. } => None,
            Source::Constant(v) => v.get(..n).map(<[u32]>::to_vec),
        }
    }

    /// The resources `program` binds, made ready before any encoder opens
    /// (uploads end the open pass).
    pub(super) fn resolve(
        &mut self,
        host: &dyn Host,
        program: &Program,
        user: &[u32],
    ) -> Option<Bound> {
        let mut bound = Bound {
            memory: None,
            gds: None,
            sampler_words: Vec::new(),
            texture_words: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            constants: Vec::new(),
            heaps: Vec::new(),
            resident: Vec::new(),
        };
        if program.translated.gds {
            bound.gds = Some(self.gds_buffer()?);
        }
        if program.translated.guest_memory {
            let m = match self.guest_window(host, user, program.address) {
                Some(m) => m,
                None => {
                    diag::problems::record(
                        diag::problems::Kind::MemoryWindow,
                        "no guest memory window, ran with memory reading 0",
                        format!("{:#x}", program.address),
                    );
                    self.report(
                        tracing::Level::ERROR,
                        format!(
                            "no guest memory window for the program at {:#x}, user data {user:x?}; it runs with memory reading 0",
                            program.address
                        ),
                    );
                    self.no_window()?
                }
            };
            let win = m.win;
            self.note(|| format!("program {:#x} windows {win:x?}", program.address));
            bound.memory = Some(m);
        }
        for r in &program.translated.resources {
            match *r {
                Resource::Texture {
                    slot,
                    source:
                        Source::Indexed {
                            ref buffer, offset, ..
                        },
                    kind,
                    view,
                    ..
                } => {
                    let index = binding::HEAPS as usize + bound.heaps.len();
                    let (buffer, at, textures) =
                        self.texture_heap(host, user, buffer, offset, kind, view)?;
                    let count = textures.len() as u32;
                    self.note(|| {
                        format!(
                            "program {:#x} t{slot} {kind:?} from a table of {count}",
                            program.address
                        )
                    });
                    bound.heaps.push((index, buffer, at));
                    bound.resident.extend(textures);
                    bound.texture_words.push((slot as usize, [count, 0]));
                }
                Resource::Texture {
                    slot,
                    ref source,
                    kind,
                    access,
                    view,
                } => {
                    let d = self.descriptor(host, user, source, 8)?;
                    let image = Image::read(&d);
                    self.note(|| {
                        format!(
                            "program {:#x} t{slot} {kind:?} {access:?} {image:x?}",
                            program.address
                        )
                    });
                    if kind == TextureKind::Cube && access != Access::Sample {
                        diag::problems::record(
                            diag::problems::Kind::UnsupportedFormat,
                            format!("{kind:?} {access:?} access"),
                            format!("{:#x}", program.address),
                        );
                        self.report(
                            tracing::Level::ERROR,
                            format!("{kind:?} textures with {access:?} access not supported"),
                        );
                        return None;
                    }
                    let tex = if access == Access::Atomic {
                        let sint = program.sint_textures >> slot & 1 != 0;
                        let want = if sint {
                            MTLPixelFormat::R32Sint
                        } else {
                            MTLPixelFormat::R32Uint
                        };
                        if image.address == 0 || image.kind < 8 {
                            None
                        } else {
                            self.atomic_image(host, image, kind).and_then(|t| {
                                if t.pixelFormat() == want {
                                    Some(t)
                                } else if atomic_capable(t.pixelFormat()) {
                                    t.newTextureViewWithPixelFormat(want)
                                } else {
                                    None
                                }
                            })
                        }
                    } else if access != Access::Sample {
                        if image.address == 0 || image.kind < 8 {
                            None
                        } else {
                            let t = self.storage_image(host, image, kind);
                            if view == View::Bits || program.coded_textures >> slot & 1 != 0 {
                                t.and_then(|t| bits_view(t, image.format))
                            } else {
                                t.and_then(|t| store_swizzled(t, image.swizzle))
                                    .and_then(without_srgb)
                            }
                        }
                    } else if image.address == 0 || image.kind < 8 {
                        self.null_texture(kind)
                    } else {
                        let coded = program.coded_textures >> slot & 1 != 0;
                        self.read_view(host, image, kind, view, coded)
                    };
                    let Some(tex) = tex else {
                        let how = match access {
                            Access::Sample => "sample",
                            Access::Write => "write",
                            Access::ReadWrite => "read-write",
                            Access::Atomic => "atomic",
                        };
                        *self
                            .image_misses
                            .entry((kind, how, image.kind, image.raw_tile, image.format))
                            .or_default() += 1;
                        let reason = if image_format(image.format).is_none() {
                            format!("image format {:#x} ({how} {kind:?})", image.format)
                        } else if image.tile.is_none() {
                            format!(
                                "tile mode {:#x} ({how} {kind:?} format {:#x})",
                                image.raw_tile, image.format
                            )
                        } else {
                            format!(
                                "image layout ({how} {kind:?} kind {:#x} tile {:#x} format {:#x})",
                                image.kind, image.raw_tile, image.format
                            )
                        };
                        diag::problems::record(
                            diag::problems::Kind::UnsupportedFormat,
                            reason,
                            format!("{:#x}", program.address),
                        );
                        self.report(
                            tracing::Level::ERROR,
                            format!(
                                "image {image:x?} for the program at {:#x} not supported",
                                program.address
                            ),
                        );
                        return None;
                    };
                    if access != Access::Sample {
                        self.stored_images.push((image, tex.clone()));
                    }
                    bound.texture_words.push((slot as usize, [d[1], d[3]]));
                    bound.textures.push((slot as usize, tex));
                }
                Resource::Constants {
                    slot,
                    ref base,
                    bytes,
                } => {
                    let at = match base {
                        Base::Pointer(p) => {
                            let d = self.descriptor(host, user, p, 2)?;
                            d[0] as u64 | (d[1] as u64) << 32
                        }
                        Base::Buffer(v) => {
                            let d = self.descriptor(host, user, v, 4)?;
                            d[0] as u64 | (d[1] as u64 & 0xFFFF) << 32
                        }
                    };
                    let (buffer, offset) = self.constants(host, at, bytes as u64)?;
                    let index = binding::CONSTANTS as usize + slot as usize;
                    bound.constants.push((index, buffer, offset));
                }
                Resource::Sampler { slot, ref source } => {
                    let d = self.descriptor(host, user, source, 4)?;
                    bound
                        .sampler_words
                        .push((slot as usize, d[..4].try_into().ok()?));
                    bound
                        .samplers
                        .push((slot as usize, self.sampler(Sampler::read(&d))?));
                }
            }
        }
        Some(bound)
    }

    /// A texture slot's view of `image` for sampling or reading; `coded`
    /// for a format the program converts.
    fn read_view(
        &mut self,
        host: &dyn Host,
        image: Image,
        kind: TextureKind,
        view: View,
        coded: bool,
    ) -> Option<Retained<Texture>> {
        if image_format(image.format).is_none() && view == View::Sampled && !coded {
            None
        } else if view == View::Bits {
            self.image(host, image, kind)
                .and_then(|t| bits_view(t, image.format))
        } else if view == View::Raw || coded {
            self.image(host, image, kind).and_then(without_srgb)
        } else {
            self.image(host, image, kind)
                .and_then(|t| self.swizzled(t, channel_selects(&image)))
        }
    }

    /// The table of texture handles for a slot whose T#s come from the
    /// buffer the V# at `buffer` describes, `phase` bytes past a multiple of
    /// 32 (see `binding::HEAPS`): its buffer and offset, and the textures in
    /// it. T#s the slot's float view cannot read (integer formats, formats
    /// the program would convert, layouts not supported) read as nothing.
    fn texture_heap(
        &mut self,
        host: &dyn Host,
        user: &[u32],
        buffer: &Source,
        phase: u32,
        kind: TextureKind,
        view: View,
    ) -> Option<(Retained<Buffer>, usize, Vec<Retained<Texture>>)> {
        let v = self.descriptor(host, user, buffer, 4)?;
        let base = v[0] as u64 | (v[1] as u64 & 0xFFFF) << 32;
        let stride = (v[1] >> 16 & 0x3FFF) as u64;
        let bytes = if stride == 0 {
            v[2] as u64
        } else {
            v[2] as u64 * stride
        };
        let entries = (bytes.saturating_sub(phase as u64) / 32) as usize;
        if entries > MAX_HEAP_ENTRIES {
            diag::problems::record(
                diag::problems::Kind::UnsupportedFormat,
                format!("table of {entries} T#s"),
                format!("{base:#x}"),
            );
        }
        let entries = entries.min(MAX_HEAP_ENTRIES);
        let mut raw = vec![0u8; entries * 32];
        if !host.read(base + phase as u64, &mut raw) {
            raw.fill(0);
        }
        let null = self.null_texture(kind)?;
        let mut textures = Vec::with_capacity(entries + 1);
        for e in raw.as_chunks::<32>().0 {
            let d: Vec<u32> = e
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            let image = Image::read(&d);
            let float = image_format(image.format).is_some_and(|f| number(f.0) == Number::Float);
            let t = if image.address == 0 || image.kind < 8 || !float {
                None
            } else {
                self.read_view(host, image, kind, view, false)
            };
            textures.push(t.unwrap_or_else(|| null.clone()));
        }
        textures.push(null);
        let (table, at) = self.scratch(8 * textures.len())?;
        // SAFETY: `scratch` gave 8 bytes per texture at `at`, which no GPU
        // work uses yet.
        let out = unsafe { table.contents().as_ptr().cast::<u8>().add(at) };
        for (i, t) in textures.iter().enumerate() {
            // SAFETY: an `MTLResourceID` is one 64-bit handle, which Metal
            // reads from an argument buffer as a texture.
            let id: u64 = unsafe { std::mem::transmute(t.gpuResourceID()) };
            // SAFETY: as above, `i` is below the textures given room for.
            unsafe { out.add(8 * i).cast::<u64>().write_unaligned(id) };
        }
        Some((table, at, textures))
    }

    pub(super) fn apply(enc: Slot, bound: &Bound, user: &[u32]) {
        let at = binding::SAMPLER_WORDS as usize;
        let mut data = [0u32; binding::USER_WORDS as usize];
        let n = user.len().min(at);
        data[..n].copy_from_slice(&user[..n]);
        for &(slot, words) in &bound.sampler_words {
            if let Some(d) = data.get_mut(at + 4 * slot..at + 4 * slot + 4) {
                d.copy_from_slice(&words);
            }
        }
        let at = binding::TEXTURE_WORDS as usize;
        for &(slot, words) in &bound.texture_words {
            if let Some(d) = data.get_mut(at + 2 * slot..at + 2 * slot + 2) {
                d.copy_from_slice(&words);
            }
        }
        enc.bytes(data.as_ptr().cast(), data.len() * 4, binding::USER as usize);
        if let Some(m) = &bound.memory {
            enc.buffer(&m.buffer, 0, binding::MEM as usize);
            if let Some(b) = &m.second {
                enc.buffer(b, 0, SECOND_WINDOW);
                enc.reach(b);
            }
            if let Some(b) = &m.third {
                enc.buffer(b, 0, THIRD_WINDOW);
                enc.reach(b);
            }
            enc.bytes(
                m.win.as_ptr().cast(),
                std::mem::size_of_val(&m.win),
                binding::MEM_BASE as usize,
            );
        }
        if let Some(gds) = &bound.gds {
            enc.buffer(gds, 0, binding::GDS as usize);
        }
        for (slot, t) in &bound.textures {
            enc.texture(t, *slot);
        }
        for (slot, s) in &bound.samplers {
            enc.sampler(s, *slot);
        }
        for (index, b, offset) in &bound.constants {
            enc.buffer(b, *offset, *index);
        }
        for (index, b, offset) in &bound.heaps {
            enc.buffer(b, *offset, *index);
        }
        for t in &bound.resident {
            enc.make_resident(ProtocolObject::from_ref(&**t), MTLResourceUsage::Read);
        }
    }

    /// The buffer and offset of a constant buffer of `len` bytes at guest address `at`.
    /// Outside guest memory, a copy in which unreadable dwords are 0, as scalar loads read them.
    fn constants(
        &mut self,
        host: &dyn Host,
        at: u64,
        len: u64,
    ) -> Option<(Retained<Buffer>, usize)> {
        if at.is_multiple_of(4)
            && let Some(b) = self.memory_buffer(host, at, len)
        {
            return Some(b);
        }
        let (buffer, offset) = self.scratch(len as usize)?;
        let words = len.div_ceil(4);
        // SAFETY: `scratch` gave `len` bytes, rounded up to dwords, at `offset`
        // that no GPU work uses yet.
        let out = unsafe {
            buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<u32>()
        };
        for i in 0..words {
            let mut w = [0u8; 4];
            let a = at.wrapping_add(4 * i);
            let mapped = host.extent(a).is_some_and(|(_, end)| a + 4 <= end);
            let v = if mapped && host.read(a, &mut w) {
                u32::from_le_bytes(w)
            } else {
                0
            };
            // SAFETY: as above, `i` is below the dwords given.
            unsafe { out.add(i as usize).write_unaligned(v) };
        }
        Some((buffer, offset))
    }
}
