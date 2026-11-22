//! Binding a program's resources to an encoder.

use super::*;

/// How much of a table the user data points to is searched for V#s.
pub(super) const TABLE_DWORDS: usize = 64;

pub(super) struct Bound {
    memory: Option<Memory>,
    pub(super) gds: Option<Retained<Buffer>>,
    /// Dword 0 of each sampler slot's S#, by slot.
    sampler_words: Vec<(usize, u32)>,
    pub(super) textures: Vec<(usize, Retained<Texture>)>,
    pub(super) samplers: Vec<(usize, Retained<ProtocolObject<dyn MTLSamplerState>>)>,
}

/// Where a program's resources are bound: one stage of an encoder.
#[derive(Clone, Copy)]
pub(super) enum Slot<'a> {
    Vertex(&'a RenderEncoder),
    Fragment(&'a RenderEncoder),
    Compute(&'a ProtocolObject<dyn MTLComputeCommandEncoder>),
}

impl Slot<'_> {
    pub(super) fn bytes(self, data: *const std::ffi::c_void, len: usize, index: usize) {
        let data = NonNull::new(data as *mut std::ffi::c_void).expect("bound data is not null");
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexBytes_length_atIndex(data, len, index),
                Slot::Fragment(e) => e.setFragmentBytes_length_atIndex(data, len, index),
                Slot::Compute(e) => e.setBytes_length_atIndex(data, len, index),
            }
        }
    }

    fn texture(self, t: &Texture, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexTexture_atIndex(Some(t), index),
                Slot::Fragment(e) => e.setFragmentTexture_atIndex(Some(t), index),
                Slot::Compute(e) => e.setTexture_atIndex(Some(t), index),
            }
        }
    }

    fn sampler(self, s: &ProtocolObject<dyn MTLSamplerState>, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexSamplerState_atIndex(Some(s), index),
                Slot::Fragment(e) => e.setFragmentSamplerState_atIndex(Some(s), index),
                Slot::Compute(e) => e.setSamplerState_atIndex(Some(s), index),
            }
        }
    }

    /// Makes `b` resident for programs that reach it only through GPU addresses.
    fn reach(self, b: &Buffer) {
        let r = ProtocolObject::from_ref(b);
        let usage = MTLResourceUsage::Read | MTLResourceUsage::Write;
        match self {
            Slot::Vertex(e) => e.useResource_usage_stages(r, usage, MTLRenderStages::Vertex),
            Slot::Fragment(e) => e.useResource_usage_stages(r, usage, MTLRenderStages::Fragment),
            Slot::Compute(e) => e.useResource_usage(r, usage),
        }
    }

    pub(super) fn buffer(self, b: &Buffer, offset: usize, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexBuffer_offset_atIndex(Some(b), offset, index),
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
            Source::Scaled { .. } => None,
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
            textures: Vec::new(),
            samplers: Vec::new(),
        };
        if program.translated.gds {
            bound.gds = Some(self.gds_buffer()?);
        }
        if program.translated.guest_memory {
            let Some(m) = self.guest_window(host, user, program.address) else {
                self.report(
                    tracing::Level::ERROR,
                    format!(
                        "no guest memory window for the program at {:#x}, user data {user:x?}",
                        program.address
                    ),
                );
                return None;
            };
            let win = m.win;
            self.note(|| format!("program {:#x} windows {win:x?}", program.address));
            bound.memory = Some(m);
        }
        for r in &program.translated.resources {
            match *r {
                Resource::Texture {
                    slot,
                    ref source,
                    kind,
                    access,
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
                            self.storage_image(host, image, kind).and_then(|t| {
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
                            self.storage_image(host, image, kind)
                                .and_then(|t| store_swizzled(t, image.swizzle))
                        }
                    } else if image.address == 0 || image.kind < 8 {
                        self.null_texture(kind)
                    } else {
                        self.image(host, image, kind)
                            .and_then(|t| self.swizzled(t, image.swizzle))
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
                        self.report(
                            tracing::Level::ERROR,
                            format!(
                                "image {image:x?} for the program at {:#x} not supported",
                                program.address
                            ),
                        );
                        return None;
                    };
                    bound.textures.push((slot as usize, tex));
                }
                Resource::Sampler { slot, ref source } => {
                    let d = self.descriptor(host, user, source, 4)?;
                    bound.sampler_words.push((slot as usize, d[0]));
                    bound
                        .samplers
                        .push((slot as usize, self.sampler(Sampler::read(&d))?));
                }
            }
        }
        Some(bound)
    }

    pub(super) fn apply(enc: Slot, bound: &Bound, user: &[u32]) {
        let at = binding::SAMPLER_WORDS as usize;
        let mut data = [0u32; 2 * binding::SAMPLER_WORDS as usize];
        let n = user.len().min(at);
        data[..n].copy_from_slice(&user[..n]);
        for &(slot, word) in &bound.sampler_words {
            if let Some(d) = data.get_mut(at + slot) {
                *d = word;
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
    }
}
