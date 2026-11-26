//! Running a case on an amdgpu device: its pages at their own GPU
//! addresses, its compute registers, one dispatch, then every lane's results.

use std::time::Duration;

use crate::amdgpu::{Buffer, Device};
use crate::{Case, PAGE, words};

/// Where the command buffer goes: high in the user range, away from test memory.
const IB_ADDRESS: u64 = 0x7F00_0000_0000;
const IB_SIZE: u64 = 64 << 10;

/// Every record's output words then the checked memory as the hardware left
/// them, or `None` if the
/// dispatch did not finish within `timeout`.
pub fn run(device: &Device, case: &Case, timeout: Duration) -> Result<Option<Vec<u32>>, String> {
    let out_len = (case.records() * case.out_words) as u64 * 4;
    let mut pages: Vec<u64> = case.input.keys().copied().collect();
    let first = case.outputs & !(PAGE - 1);
    pages.extend((first..case.outputs + out_len).step_by(PAGE as usize));
    let (checked, checked_len) = (case.checked.0, case.checked.1 as u64);
    pages.extend((checked & !(PAGE - 1)..checked + checked_len).step_by(PAGE as usize));
    pages.sort_unstable();
    pages.dedup();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for p in pages {
        match runs.last_mut() {
            Some((_, end)) if *end == p => *end += PAGE,
            _ => runs.push((p, p + PAGE)),
        }
    }
    if runs
        .iter()
        .any(|&(lo, hi)| lo < IB_ADDRESS + IB_SIZE && IB_ADDRESS < hi)
    {
        return Err(format!(
            "the case uses {IB_ADDRESS:#x}, where the command buffer goes"
        ));
    }

    let mut buffers: Vec<Buffer> = Vec::with_capacity(runs.len());
    for &(start, end) in &runs {
        let mut b = device.buffer(start, end - start)?;
        let bytes = b.bytes_mut();
        for (i, page) in (start..end).step_by(PAGE as usize).enumerate() {
            let dst = &mut bytes[i * PAGE as usize..(i + 1) * PAGE as usize];
            match case.input.get(&page) {
                Some(src) => dst.copy_from_slice(src),
                None => dst.fill(0),
            }
        }
        buffers.push(b);
    }

    let commands: Vec<u32> = case
        .preamble
        .iter()
        .copied()
        .chain(case.commands())
        .collect();
    let mut commands = commands;
    while !commands.len().is_multiple_of(8) {
        commands.push((3 << 30) | (0x3FFF << 16) | (0x10 << 8));
    }
    let ip = if case.graphics {
        crate::amdgpu::HW_IP_GFX
    } else {
        crate::amdgpu::HW_IP_COMPUTE
    };
    let mut ib = device.buffer(IB_ADDRESS, IB_SIZE)?;
    for (dst, w) in ib
        .bytes_mut()
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(&commands)
    {
        *dst = w.to_le_bytes();
    }
    let resident: Vec<&Buffer> = buffers.iter().collect();
    if !device.run(ip, &ib, commands.len() as u32, &resident, timeout)? {
        return Ok(None);
    }

    let read = |addr: u64, len: u64| -> Vec<u32> {
        let b = buffers
            .iter()
            .find(|b| b.addr <= addr && addr + len <= b.addr + b.size)
            .expect("compared memory is mapped in one run");
        let at = (addr - b.addr) as usize;
        words(&b.bytes()[at..at + len as usize])
    };
    let mut out = if out_len > 0 {
        read(case.outputs, out_len)
    } else {
        Vec::new()
    };
    if checked_len > 0 {
        out.extend(read(checked, checked_len));
    }
    Ok(Some(out))
}
