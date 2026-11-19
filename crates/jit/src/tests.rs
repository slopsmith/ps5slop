use std::sync::Mutex;

use a64::{Asm, Size, Width, r};
use memory::{PAGE_SIZE, Prot};
use x86::cpu::flags;
use x86::cpu::reg::*;

use super::*;

const RW: u64 = 0x40_0000;
const RO: u64 = 0x50_0000;

/// The VM is process-wide, so tests that use it take turns.
static VM: Mutex<()> = Mutex::new(());

/// A guest address space with a read-write page at `RW` holding 41 and a
/// read-only page at `RO`.
fn guest() -> Memory {
    let mem = Memory::new(1 << 40).unwrap();
    mem.map(RW, PAGE_SIZE, Prot::RW, "rw").unwrap();
    mem.map(RO, PAGE_SIZE, Prot::READ, "ro").unwrap();
    mem.write_u64(RW, 41).unwrap();
    mem
}

/// `x1 = [x0]; x1 += 1; [x0] = x1; hvc #7`.
fn increment() -> Vec<u32> {
    let mut a = Asm::new();
    a.ldr(Width::X, r(1), r(0), 0);
    a.add_imm(Size::X, r(1), r(1), 1);
    a.str(Width::X, r(1), r(0), 0);
    a.hvc(7);
    a.finish()
}

#[test]
fn code_in_the_vm_reads_and_writes_guest_memory_at_its_own_address() {
    let _vm = VM.lock().unwrap_or_else(|e| e.into_inner());
    let mem = guest();
    let mut m = machine().unwrap();
    assert!(m.sync(&mem).unwrap());
    let pc = m.load_code(0x1000, &increment());
    let mut cpu = Cpu::new(&mut m).unwrap().unwrap();
    assert!(cpu.tso().unwrap());
    cpu.set_reg(0, RW).unwrap();
    assert_eq!(cpu.run(&m, pc).unwrap(), Exit::Hvc(7));
    assert_eq!(mem.read_u64(RW).unwrap(), 42);
    cpu.set_reg(0, RO).unwrap();
    let Exit::Fault { esr, far, .. } = cpu.run(&m, pc).unwrap() else {
        panic!("a write to a read-only page must fault")
    };
    assert_eq!((esr >> 26, far), (0x25, RO));
    cpu.set_reg(0, 0x60_0000).unwrap();
    assert!(matches!(
        cpu.run(&m, pc).unwrap(),
        Exit::Fault { far: 0x60_0000, .. }
    ));
    cpu.release(&mut m);
}

#[test]
fn mapping_changes_reach_the_vm_after_a_sync() {
    let _vm = VM.lock().unwrap_or_else(|e| e.into_inner());
    let mem = guest();
    let mut m = machine().unwrap();
    m.sync(&mem).unwrap();
    let pc = m.load_code(0x1000, &increment());
    let mut cpu = Cpu::new(&mut m).unwrap().unwrap();
    mem.map(0x60_0000, PAGE_SIZE, Prot::RW, "new").unwrap();
    assert!(m.sync(&mem).unwrap());
    cpu.set_reg(0, 0x60_0000).unwrap();
    assert_eq!(cpu.run(&m, pc).unwrap(), Exit::Hvc(7));
    assert_eq!(mem.read_u64(0x60_0000).unwrap(), 1);
    cpu.release(&mut m);
}

#[test]
fn memory_mapped_again_with_the_same_bounds_reaches_the_vm() {
    let _vm = VM.lock().unwrap_or_else(|e| e.into_inner());
    let mem = guest();
    let mut m = machine().unwrap();
    mem.map(0x60_0000, PAGE_SIZE, Prot::RW, "first").unwrap();
    m.sync(&mem).unwrap();
    let pc = m.load_code(0x1000, &increment());
    let mut cpu = Cpu::new(&mut m).unwrap().unwrap();
    mem.map_over(0x60_0000, PAGE_SIZE, Prot::RW, "second").unwrap();
    assert!(m.sync(&mem).unwrap());
    cpu.set_reg(0, 0x60_0000).unwrap();
    assert_eq!(cpu.run(&m, pc).unwrap(), Exit::Hvc(7));
    assert_eq!(mem.read_u64(0x60_0000).unwrap(), 1);
    cpu.release(&mut m);
}

const CODE: u64 = 0x40_0000;
const STACK: u64 = 0x10_0000;
const TRAP: u64 = 0xff_0000_0000;

/// An address space with `code` at `CODE` (one writable page, so the code
/// can keep data next to it), a stack and a trap region, and a CPU at its start.
fn program(code: &[u8]) -> (Memory, x86::Cpu) {
    let mem = Memory::new(1 << 40).unwrap();
    mem.map(CODE, PAGE_SIZE, Prot::new(true, true, true), "code")
        .unwrap();
    mem.write_host(CODE, code).unwrap();
    mem.map(STACK, PAGE_SIZE, Prot::RW, "stack").unwrap();
    mem.map_trap(TRAP, PAGE_SIZE, "trap").unwrap();
    let mut cpu = x86::Cpu {
        rip: CODE,
        ..x86::Cpu::default()
    };
    cpu.regs[RSP] = STACK + PAGE_SIZE;
    cpu.regs[RBP] = TRAP;
    (mem, cpu)
}

fn page(mem: &Memory) -> Vec<u8> {
    let mut b = vec![0u8; PAGE_SIZE as usize];
    mem.read(CODE, &mut b).unwrap();
    b
}

/// Every instruction form the translator handles, plus two it leaves to the
/// interpreter (ADC, three-operand IMUL); ends with `call rbp`.
const MIXED: &[u8] = &[
    0xb9, 0x64, 0x00, 0x00, 0x00, 0x31, 0xc0, 0x48, 0x8d, 0x1d, 0x00, 0x02, 0x00, 0x00, 0x01, 0xc8,
    0x89, 0x04, 0x8b, 0xff, 0xc9, 0x75, 0xf7, 0x48, 0xba, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34,
    0x12, 0x49, 0x89, 0xd0, 0x49, 0xc1, 0xe0, 0x04, 0x48, 0xc1, 0xea, 0x03, 0x49, 0xd1, 0xf8, 0x80,
    0xc2, 0xf0, 0x80, 0xee, 0x7f, 0x45, 0x31, 0xc9, 0x3d, 0xba, 0x13, 0x00, 0x00, 0x41, 0x0f, 0x94,
    0xc1, 0x4c, 0x6b, 0xd2, 0x01, 0x4c, 0x0f, 0xaf, 0xd2, 0x41, 0xbb, 0x07, 0x00, 0x00, 0x00, 0x41,
    0x83, 0xfb, 0x09, 0x4c, 0x0f, 0x4c, 0xd8, 0x49, 0xff, 0xc3, 0x44, 0x0f, 0xb6, 0x63, 0x08, 0x4c,
    0x0f, 0xbf, 0x6b, 0x08, 0x41, 0x54, 0x6a, 0x55, 0x41, 0x5e, 0x41, 0x5f, 0xe8, 0x10, 0x00, 0x00,
    0x00, 0x66, 0x41, 0x81, 0xe4, 0x0f, 0xff, 0x4c, 0x09, 0xc8, 0xa9, 0x40, 0x00, 0x00, 0x00, 0xff,
    0xd5, 0x49, 0x83, 0xc5, 0x01, 0x83, 0xd0, 0x00, 0xc3,
];

#[test]
fn translated_code_ends_in_the_same_state_as_the_interpreter() {
    let _vm = VM.lock().unwrap_or_else(|e| e.into_inner());
    let (mem, mut want) = program(MIXED);
    assert_eq!(x86::run(&mut want, &mem, 100_000), x86::Exit::Trap(TRAP));
    let expected_page = page(&mem);

    let (mem, mut got) = program(MIXED);
    assert_eq!(run(&mut got, &mem, &|| false), x86::Exit::Trap(TRAP));
    assert_eq!(got.regs, want.regs);
    assert_eq!(got.rip, want.rip);
    assert_eq!(got.rflags & flags::ARITH, want.rflags & flags::ARITH);
    assert!(page(&mem) == expected_page, "memory differs");
}

/// Calls the native `memset` (trap address +0x10) and then `memmove` (+0x20)
/// twice, overlapping backwards and forwards; ends with `call rbp`.
const NATIVES: &[u8] = &[
    0x48, 0x8d, 0x3d, 0x00, 0x04, 0x00, 0x00, 0xbe, 0xab, 0x01, 0x00, 0x00, 0xba, 0x25, 0x00, 0x00,
    0x00, 0x48, 0xb8, 0x10, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0xff, 0xd0, 0x48, 0x8d, 0x3d,
    0xe0, 0x03, 0x00, 0x00, 0x48, 0x8d, 0x35, 0xd4, 0x03, 0x00, 0x00, 0xba, 0x15, 0x00, 0x00, 0x00,
    0x48, 0xb8, 0x20, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0xff, 0xd0, 0x48, 0x8d, 0x3d, 0xc0,
    0x03, 0x00, 0x00, 0x48, 0x8d, 0x35, 0xcc, 0x03, 0x00, 0x00, 0xba, 0x13, 0x00, 0x00, 0x00, 0x48,
    0xb8, 0x20, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0xff, 0xd0, 0xff, 0xd5,
];

#[test]
fn native_memset_and_memmove_run_inside_the_vm() {
    let _vm = VM.lock().unwrap_or_else(|e| e.into_inner());
    let (mem, mut cpu) = program(NATIVES);
    let fill: Vec<u8> = (0..0x200u32).map(|k| (k * 7 + 3) as u8).collect();
    mem.write_host(CODE + 0x300, &fill).unwrap();
    set_native(TRAP + 0x10, Native::Memset);
    set_native(TRAP + 0x20, Native::Memmove);
    assert_eq!(run(&mut cpu, &mem, &|| false), x86::Exit::Trap(TRAP));

    let mut want = vec![0u8; PAGE_SIZE as usize];
    want[..NATIVES.len()].copy_from_slice(NATIVES);
    want[0x300..0x500].copy_from_slice(&fill);
    want[0x407..0x407 + 37].fill(0xab);
    want.copy_within(0x3ff..0x3ff + 21, 0x404);
    want.copy_within(0x416..0x416 + 19, 0x403);
    assert!(page(&mem) == want, "memory differs");
    assert_eq!(cpu.regs[RAX], CODE + 0x403);
}
