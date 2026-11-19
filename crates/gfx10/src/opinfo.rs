//! Opcode table lookups.

use std::sync::OnceLock;

use crate::opcodes::OPS;

/// Microcode format an opcode number belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fmt {
    Sop1,
    Sop2,
    Sopk,
    Sopp,
    Sopc,
    Smem,
    Ds,
    Mtbuf,
    Mubuf,
    Mimg,
    Exp,
    Flat,
    Global,
    Scratch,
    Vintrp,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vop3p,
}

const FMT_COUNT: usize = 20;

/// Value type of an operand or result, as ACO describes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    /// A lane mask: one bit per lane, 64 bits in wave64.
    Lane,
    Exec,
    Scc,
    M0,
    /// A 32-bit literal that follows the instruction.
    Imm,
    U16,
    U32,
    U64,
    U128,
    I64,
    F16,
    F32,
    F64,
    PkU16,
    PkF16,
}

impl Ty {
    /// Size in dwords when held in registers (`wave64` sizes lane masks).
    pub fn dwords(self, wave64: bool) -> u8 {
        match self {
            Ty::Lane | Ty::Exec => 1 + wave64 as u8,
            Ty::U64 | Ty::I64 | Ty::F64 => 2,
            Ty::U128 => 4,
            _ => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Ty::F16 | Ty::F32 | Ty::F64 | Ty::PkF16)
    }

    /// Bit width of one element (16, 32 or 64).
    pub fn bits(self) -> u8 {
        match self {
            Ty::U16 | Ty::F16 | Ty::PkU16 | Ty::PkF16 => 16,
            Ty::U64 | Ty::I64 | Ty::F64 => 64,
            _ => 32,
        }
    }
}

/// One opcode: its name in LLVM's syntax and the types of its results and operands.
#[derive(Debug)]
pub struct OpInfo {
    pub fmt: Fmt,
    pub code: u16,
    pub name: &'static str,
    pub defs: &'static [Ty],
    pub srcs: &'static [Ty],
}

const CODES: usize = 1024;

fn table() -> &'static [Option<u16>] {
    static TABLE: OnceLock<Vec<Option<u16>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![None; FMT_COUNT * CODES];
        for (i, op) in OPS.iter().enumerate() {
            t[op.fmt as usize * CODES + op.code as usize] = Some(i as u16);
        }
        t
    })
}

/// Finds the opcode `code` of format `fmt`.
pub fn lookup(fmt: Fmt, code: u16) -> Option<&'static OpInfo> {
    if code as usize >= CODES {
        return None;
    }
    table()[fmt as usize * CODES + code as usize].map(|i| &OPS[i as usize])
}

/// Finds an opcode by its LLVM name.
pub fn by_name(name: &str) -> Option<&'static OpInfo> {
    OPS.iter().find(|op| op.name == name)
}

/// All known opcodes.
pub fn all() -> &'static [OpInfo] {
    OPS
}
