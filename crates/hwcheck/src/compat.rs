//! Making a gfx1013 program run on the GPU at hand, and checking that it
//! means the same there.
//!
//! The PS5's ISA is GFX10.1. GFX10.3 (RDNA2) dropped the unfused
//! multiply-adds (`v_mad_f32`, `v_mac_f32`, `v_madmk_f32`, `v_madak_f32`)
//! and reuses their opcodes, and turned the legacy ones into fused forms
//! under the same opcodes. [`port`] rewrites the unfused ones to their fused
//! twins; results may then differ from the PS5 in the last bit. llvm-mc then
//! disassembles the original for gfx1013 and the rewrite for the target; any
//! other difference means a hardware run is not evidence.

use std::io::Write;
use std::process::{Command, Stdio};

use crate::Memory;

/// The program's words, from its start through its last instruction.
pub fn program_words(code: &[u32]) -> Result<&[u32], String> {
    let program = gfx10::translate::program(code).map_err(|e| format!("{e:?}"))?;
    let end = program
        .last()
        .map_or(0, |(pc, inst)| pc + inst.words as usize);
    Ok(&code[..end.min(code.len())])
}

const VOP2_REWRITES: [(u32, u32); 3] = [(0x1F, 0x2B), (0x20, 0x2C), (0x21, 0x2D)];
const VOP3_REWRITES: [(u32, u32); 2] = [(0x141, 0x14B), (0x11F, 0x12B)];

/// The names gfx1013 code has and `cpu` knows under another name, with that name.
fn renames(cpu: &str) -> &'static [(&'static str, &'static str)] {
    if cpu.starts_with("gfx103") {
        &[
            ("v_mad_legacy_f32", "v_fma_legacy_f32"),
            ("v_mac_legacy_f32", "v_fmac_legacy_f32"),
            ("v_mad_f32", "v_fma_f32"),
            ("v_mac_f32", "v_fmac_f32"),
            ("v_madmk_f32", "v_fmamk_f32"),
            ("v_madak_f32", "v_fmaak_f32"),
        ]
    } else {
        &[]
    }
}

/// `words` rewritten for `cpu`, with the number of instructions that run fused there.
pub fn port(words: &[u32], cpu: &str) -> Result<(Vec<u32>, usize), String> {
    let mut out = words.to_vec();
    if renames(cpu).is_empty() {
        return Ok((out, 0));
    }
    let program = gfx10::translate::program(words).map_err(|e| format!("{e:?}"))?;
    let mut fused = 0;
    for (pc, inst) in program {
        let w = &mut out[pc];
        let name = inst.op.name;
        match inst.op.fmt {
            gfx10::opinfo::Fmt::Vop2 => {
                let op = *w >> 25 & 0x3F;
                if let Some(&(_, to)) = VOP2_REWRITES.iter().find(|r| r.0 == op) {
                    *w = (*w & !(0x3F << 25)) | to << 25;
                    fused += 1;
                } else if name == "v_mac_legacy_f32" {
                    fused += 1;
                }
            }
            gfx10::opinfo::Fmt::Vop3 => {
                let op = *w >> 16 & 0x3FF;
                if let Some(&(_, to)) = VOP3_REWRITES.iter().find(|r| r.0 == op) {
                    *w = (*w & !(0x3FF << 16)) | to << 16;
                    fused += 1;
                } else if name == "v_mad_legacy_f32" || name == "v_mac_legacy_f32" {
                    fused += 1;
                }
            }
            _ => {}
        }
    }
    Ok((out, fused))
}

fn disassemble(words: &[u32], cpu: &str) -> Result<Vec<String>, String> {
    let mut input = String::new();
    for w in words {
        for b in w.to_le_bytes() {
            input.push_str(&format!("{b:#04x} "));
        }
        input.push('\n');
    }
    let mut child = Command::new("llvm-mc")
        .args(["--disassemble", "-arch=amdgcn", &format!("-mcpu={cpu}")])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run llvm-mc ({e}); run inside `nix develop`"))?;
    child
        .stdin
        .take()
        .expect("piped")
        .write_all(input.as_bytes())
        .map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("invalid instruction encoding") {
        return Err(format!("{cpu} has no encoding for some instructions"));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.split("//").next().unwrap_or_default().trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('.'))
        .collect())
}

/// Instructions whose disassembly differs between `original` on gfx1013
/// and `ported` on `cpu`, beyond the renames [`port`] makes, as text pairs.
pub fn differences(
    original: &[u32],
    ported: &[u32],
    cpu: &str,
) -> Result<Vec<(String, String)>, String> {
    let ps5 = disassemble(original, "gfx1013")?;
    let other = disassemble(ported, cpu)?;
    let mut out = Vec::new();
    for i in 0..ps5.len().max(other.len()) {
        let mut a = ps5.get(i).cloned().unwrap_or_default();
        for (from, to) in renames(cpu) {
            if a.starts_with(from) {
                a = a.replacen(from, to, 1);
                break;
            }
        }
        let b = other.get(i).cloned().unwrap_or_default();
        if a != b {
            out.push((a, b));
        }
    }
    Ok(out)
}

/// What porting a program to `cpu` did.
pub struct Ported {
    /// Instructions that run fused on `cpu` but unfused on the PS5.
    pub fused: usize,
    /// Instructions that still read differently, as `(gfx1013, cpu)` text.
    pub differences: Vec<(String, String)>,
}

/// Rewrites the program at `addr` in `mem` for `cpu` and checks the result.
pub fn port_program(mem: &mut impl Memory, addr: u64, cpu: &str) -> Result<Ported, String> {
    let code = mem.code(addr);
    let original = program_words(&code)?;
    let (ported, fused) = port(original, cpu)?;
    let differences = differences(original, &ported, cpu)?;
    let bytes: Vec<u8> = ported.iter().flat_map(|w| w.to_le_bytes()).collect();
    mem.write(addr, &bytes);
    Ok(Ported { fused, differences })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unfused_multiply_adds_become_fused_for_gfx103() {
        let mad = [0xD541_0000u32, 0x0402_0501, 0xBF81_0000];
        let (out, fused) = port(&mad, "gfx1033").unwrap();
        assert_eq!(out[0], 0xD54B_0000);
        assert_eq!(&out[1..], &mad[1..]);
        assert_eq!(fused, 1);

        let mac = [0x3E00_0300u32, 0xBF81_0000];
        let (out, fused) = port(&mac, "gfx1033").unwrap();
        assert_eq!(out[0] >> 25, 0x2B);
        assert_eq!(fused, 1);

        let (out, fused) = port(&mac, "gfx1010").unwrap();
        assert_eq!(out, mac);
        assert_eq!(fused, 0);
    }
}
