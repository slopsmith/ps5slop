//! Architectural state of one guest thread.

pub mod flags {
    pub const CF: u64 = 1 << 0;
    pub const PF: u64 = 1 << 2;
    pub const AF: u64 = 1 << 4;
    pub const ZF: u64 = 1 << 6;
    pub const SF: u64 = 1 << 7;
    pub const TF: u64 = 1 << 8;
    pub const IF: u64 = 1 << 9;
    pub const DF: u64 = 1 << 10;
    pub const OF: u64 = 1 << 11;
    pub const AC: u64 = 1 << 18;
    pub const ID: u64 = 1 << 21;

    pub const ARITH: u64 = CF | PF | AF | ZF | SF | OF;
    /// What user code may change with POPF.
    pub const USER_WRITABLE: u64 = ARITH | DF | AC | ID | TF;
    /// Bit 1 always reads as one; IF is set in user mode.
    pub const FIXED: u64 = 1 << 1 | IF;
}

pub mod reg {
    pub const RAX: usize = 0;
    pub const RCX: usize = 1;
    pub const RDX: usize = 2;
    pub const RBX: usize = 3;
    pub const RSP: usize = 4;
    pub const RBP: usize = 5;
    pub const RSI: usize = 6;
    pub const RDI: usize = 7;
    pub const R8: usize = 8;
    pub const R9: usize = 9;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cpu {
    pub regs: [u64; 16],
    pub rip: u64,
    pub rflags: u64,
    pub fs_base: u64,
    pub gs_base: u64,
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            regs: [0; 16],
            rip: 0,
            rflags: flags::FIXED,
            fs_base: 0,
            gs_base: 0,
        }
    }
}

impl Cpu {
    pub fn flag(&self, f: u64) -> bool {
        self.rflags & f != 0
    }

    pub fn set_flag(&mut self, f: u64, on: bool) {
        if on {
            self.rflags |= f;
        } else {
            self.rflags &= !f;
        }
    }

    /// Replaces the six arithmetic flags.
    pub fn set_arith(&mut self, f: u64) {
        self.rflags = (self.rflags & !flags::ARITH) | (f & flags::ARITH);
    }

    /// Condition code `cc` (the low nibble of Jcc, SETcc and CMOVcc).
    pub fn condition(&self, cc: u8) -> bool {
        use flags::*;
        let r = match cc >> 1 {
            0 => self.flag(OF),
            1 => self.flag(CF),
            2 => self.flag(ZF),
            3 => self.flag(CF) || self.flag(ZF),
            4 => self.flag(SF),
            5 => self.flag(PF),
            6 => self.flag(SF) != self.flag(OF),
            _ => self.flag(ZF) || self.flag(SF) != self.flag(OF),
        };
        r != (cc & 1 != 0)
    }
}
