//! What the renderer asks of the shader translator, and what it gets back.

/// The pipeline stage a guest program runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// The NGG program in the ES slot, or a legacy vertex shader.
    Vertex,
    Pixel,
    Compute,
}

/// A guest program and the state its translation depends on.
pub struct Request<'a> {
    pub stage: Stage,
    /// Code from the program's start; the program ends at its `s_endpgm`.
    pub code: &'a [u32],
    pub user_sgprs: u32,
    pub ps_input_ena: u32,
    pub ps_input_addr: u32,
    pub ps_input_cntl: &'a [u32],
    pub col_format: u32,
}

/// Where a resource descriptor's dwords are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    UserSgpr(u8),
    /// At `offset` bytes past the 64-bit pointer in user SGPRs `sgpr`, `sgpr + 1`.
    Memory { sgpr: u8, offset: u32 },
}

/// A resource the translated program binds, at a Metal slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resource {
    Texture { slot: u8, source: Source },
    Sampler { slot: u8, source: Source },
    Buffer { slot: u8, source: Source },
}

/// A translated program: MSL source and the bindings it expects. User
/// SGPRs arrive at `buffer(0)`, buffer `n` at `buffer(1 + n)`, the V#s of
/// all buffers at `buffer(30)`, texture and sampler `n` at index `n`.
pub struct Translated {
    pub msl: String,
    pub entry: String,
    pub resources: Vec<Resource>,
    pub vertex_outputs: u32,
    pub color_outputs: u8,
}

pub fn translate(_request: &Request) -> Result<Translated, String> {
    Err("no GFX10 shader translator yet".into())
}
