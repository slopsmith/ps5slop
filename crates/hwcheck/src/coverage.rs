//! Which gfx1013 opcodes the translator takes: one instance of each opcode
//! is built, decoded and handed to the translator.

use gfx10::decode::decode;
use gfx10::opinfo::{Fmt, OpInfo, all};
use gfx10::translate::{Request, Stage, translate};

/// What the translator does with an opcode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coverage {
    Translated,
    /// It refuses the opcode, with its reason.
    Untranslated(String),
    /// An opcode the scan has no instance of, so its coverage is open.
    Unknown,
}

const S_ENDPGM: u32 = 0xBF81_0000;

/// Words that decode as `op`: its opcode in its format's fields, other
/// fields zero (VGPR 0 where a vector operand is needed), and zero words
/// after it for a literal.
pub fn instance(op: &OpInfo) -> Option<Vec<u32>> {
    let c = op.code as u32;
    const V0: u32 = 256;
    let mut words = match op.fmt {
        Fmt::Sop2 => vec![0x8000_0000 | c << 23],
        Fmt::Sopk => vec![0xB000_0000 | c << 23],
        Fmt::Sop1 => vec![0xBE80_0000 | c << 8],
        Fmt::Sopc => vec![0xBF00_0000 | c << 16],
        Fmt::Sopp => vec![0xBF80_0000 | c << 16],
        Fmt::Smem => vec![0xF400_0000 | c << 18, 0],
        Fmt::Vop1 => vec![0x7E00_0000 | c << 9 | V0],
        Fmt::Vop2 => vec![c << 25 | V0],
        Fmt::Vopc => vec![0x7C00_0000 | c << 17 | V0],
        Fmt::Vop3 => vec![0xD400_0000 | c << 16, V0],
        Fmt::Vop3p => vec![0xCC00_0000 | c << 16, V0],
        Fmt::Vintrp => vec![0xC800_0000 | c << 16],
        Fmt::Ds => vec![0xD800_0000 | c << 18, 0],
        Fmt::Mubuf => vec![0xE000_0000 | (c & 0x7F) << 18 | (c >> 7) << 25, 0],
        Fmt::Mtbuf => vec![0xE800_0000 | (c & 7) << 16, (c >> 3) << 21],
        Fmt::Mimg => vec![0xF000_0000 | (c & 0x7F) << 18 | c >> 7, 0],
        Fmt::Exp => vec![0xF800_0000, 0],
        Fmt::Flat => vec![0xDC00_0000 | c << 18, 0],
        Fmt::Scratch => vec![0xDC00_4000 | c << 18, 0],
        Fmt::Global => vec![0xDC00_8000 | c << 18, 0],
    };
    words.extend([0, 0]);
    let inst = decode(&words).ok()?;
    if inst.op.name != op.name {
        return None;
    }
    words.truncate(inst.words as usize);
    Some(words)
}

/// What the translator does with `op`.
pub fn coverage(op: &OpInfo) -> Coverage {
    let Some(mut code) = instance(op) else {
        return Coverage::Unknown;
    };
    code.push(S_ENDPGM);
    let stage = if matches!(op.fmt, Fmt::Exp | Fmt::Vintrp) {
        Stage::Pixel
    } else {
        Stage::Compute
    };
    let req = Request {
        stage,
        code: &code,
        address: 0x10_0000,
        user_sgprs: 4,
        ngg: false,
        ps_input_ena: 1,
        ps_input_addr: 1,
        ps_input_cntl: &[0; 32],
        col_format: 0x4,
        compute_rsrc2: 4 << 1 | 1 << 7,
        threads: [64, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
        grid_bounds: false,
        drop_depth_export: false,
        half_targets: 0,
        logic_op: 0xCC,
        logic_targets: [0; 8],
        swizzles: [0; 8],
        probe_flat: false,
    };
    match translate(&req) {
        Err(e) if e.message.contains("unsupported instruction") => {
            Coverage::Untranslated(e.message)
        }
        _ => Coverage::Translated,
    }
}

/// Why `cpu` cannot test `op` (it lacks the opcode, runs it differently from
/// gfx1013, or llvm-mc cannot tell), or `None` if it can.
pub fn untestable(op: &OpInfo, cpu: &str) -> Option<String> {
    let reason = match op.name {
        "v_illegal" | "s_trap" => Some("raises an exception that ends the program"),
        "s_code_end" => Some("padding after a program, never run"),
        "s_sendmsg" | "s_sendmsghalt" => {
            Some("a message to fixed-function units, with no result of its own")
        }
        _ => None,
    };
    if let Some(r) = reason {
        return Some(r.into());
    }
    let mut words = instance(op)?;
    words.push(S_ENDPGM);
    match crate::compat::port(&words, cpu) {
        Err(e) => Some(e),
        Ok((_, fused)) if fused > 0 => Some(format!("{cpu} runs it fused, gfx1013 unfused")),
        Ok((ported, _)) => match crate::compat::differences(&words, &ported, cpu) {
            Err(e) if e.starts_with(cpu) => Some(e),
            Err(_) => None,
            Ok(d) if d.is_empty() => None,
            Ok(d) => Some(format!("{cpu} reads {} as {}", d[0].0, d[0].1)),
        },
    }
}

/// An encoding of a VOP1, VOP2 or VOPC opcode besides its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Promotion {
    Vop3,
    Dpp,
    Sdwa,
}

/// Whether `op` has the encoding `p`, as LLVM's disassembler accepts it.
pub fn has(op: &OpInfo, p: Promotion) -> bool {
    use gfx10::print::{Syntax, print_checked};
    if !matches!(op.fmt, Fmt::Vop1 | Fmt::Vop2 | Fmt::Vopc) {
        return false;
    }
    let base = match op.fmt {
        Fmt::Vop1 => 0x7E00_0000 | (op.code as u32) << 9,
        Fmt::Vop2 => (op.code as u32) << 25,
        _ => 0x7C00_0000 | (op.code as u32) << 17,
    };
    let words = match p {
        Promotion::Vop3 => {
            let code = match op.fmt {
                Fmt::Vopc => op.code,
                Fmt::Vop2 => op.code + 0x100,
                _ => op.code + 0x180,
            };
            return gfx10::decode::vop3_op(code).is_some_and(|o| o.name == op.name);
        }
        Promotion::Dpp => {
            const QUAD_PERM_IDENTITY: u32 = 0xE4 << 8;
            vec![base | 0xFA, QUAD_PERM_IDENTITY | 0xF << 24 | 0xF << 28]
        }
        Promotion::Sdwa => {
            const DWORD: u32 = 6;
            let src1 = if op.fmt == Fmt::Vop1 { 0 } else { DWORD << 24 };
            let dst = if op.fmt == Fmt::Vopc { 0 } else { DWORD << 8 };
            vec![base | 0xF9, dst | DWORD << 16 | src1]
        }
    };
    decode(&words)
        .ok()
        .filter(|i| i.op.name == op.name)
        .and_then(|i| print_checked(&i, Syntax { wave64: true }))
        .is_some()
}

/// Every opcode in the table with what the translator does with it.
pub fn scan() -> Vec<(&'static OpInfo, Coverage)> {
    all().iter().map(|op| (op, coverage(op))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_instance_of_almost_every_opcode_can_be_built() {
        let unknown: Vec<&str> = all()
            .iter()
            .filter(|op| instance(op).is_none())
            .map(|op| op.name)
            .collect();
        assert!(
            unknown.len() < 10,
            "the chart would show these as unknown: {unknown:?}"
        );
    }
}
