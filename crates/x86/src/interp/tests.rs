use memory::{Access, Memory, PAGE_SIZE, Prot};

use super::*;

const CODE: u64 = 0x1_0000;
const STACK_TOP: u64 = 0x2_4000;
const DATA: u64 = 0x3_0000;
const TRAP: u64 = 0x1000_0000;

/// A machine with `code` at [`CODE`], a stack, a data page and a trap page.
/// `_asm` is the source the bytes were assembled from with llvm-mc.
fn machine(_asm: &str, code: &[u8]) -> (Cpu, Memory) {
    let mem = Memory::new(1 << 32).unwrap();
    mem.map(CODE, PAGE_SIZE, Prot::RX, "code").unwrap();
    mem.write_host(CODE, code).unwrap();
    mem.map(STACK_TOP - PAGE_SIZE, PAGE_SIZE, Prot::RW, "stack")
        .unwrap();
    mem.map(DATA, PAGE_SIZE, Prot::RW, "data").unwrap();
    mem.map_trap(TRAP, PAGE_SIZE, "trap").unwrap();
    let cpu = Cpu {
        rip: CODE,
        regs: {
            let mut r = [0; 16];
            r[RSP] = STACK_TOP;
            r
        },
        ..Cpu::default()
    };
    (cpu, mem)
}

/// Runs exactly `n` instructions, each of which must complete.
fn steps(cpu: &mut Cpu, mem: &Memory, n: usize) {
    for k in 0..n {
        let rip = cpu.rip;
        assert_eq!(
            step(cpu, mem),
            Exit::Continue,
            "instruction {k} at {rip:#x}"
        );
    }
}

#[test]
fn add_wraps_and_sets_flags() {
    let (mut cpu, mem) = machine(
        "mov eax, -1; add eax, 1",
        &[0xb8, 0xff, 0xff, 0xff, 0xff, 0x83, 0xc0, 0x01],
    );
    steps(&mut cpu, &mem, 2);
    assert_eq!(cpu.regs[RAX], 0);
    assert!(cpu.flag(ZF) && cpu.flag(CF) && !cpu.flag(OF));
}

#[test]
fn partial_register_writes_follow_long_mode_rules() {
    let (mut cpu, mem) = machine(
        "mov rax, -1; mov eax, 5; mov bx, 7; mov ah, 0x12; mov sil, 0x34",
        &[
            0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, 0xb8, 0x05, 0x00, 0x00, 0x00, 0x66, 0xbb,
            0x07, 0x00, 0xb4, 0x12, 0x40, 0xb6, 0x34,
        ],
    );
    cpu.regs[RBX] = 0xaaaa_aaaa_aaaa_aaaa;
    cpu.regs[RSI] = 0xffff;
    steps(&mut cpu, &mem, 5);
    assert_eq!(cpu.regs[RAX], 0x1205);
    assert_eq!(cpu.regs[RBX], 0xaaaa_aaaa_aaaa_0007);
    assert_eq!(cpu.regs[RSI], 0xff34);
}

#[test]
fn loops_until_the_counter_reaches_zero() {
    let (mut cpu, mem) = machine(
        "xor ecx, ecx; mov eax, 10; l: add ecx, eax; dec eax; jnz l",
        &[
            0x31, 0xc9, 0xb8, 0x0a, 0x00, 0x00, 0x00, 0x01, 0xc1, 0xff, 0xc8, 0x75, 0xfa,
        ],
    );
    steps(&mut cpu, &mem, 2 + 10 * 3);
    assert_eq!(cpu.regs[RCX], 55);
    assert_eq!(cpu.rip, CODE + 13);
}

#[test]
fn push_pop_and_call_use_the_stack() {
    let (mut cpu, mem) = machine(
        "push 0x1234; pop rbx; call t; int3; t: pop rdx",
        &[
            0x68, 0x34, 0x12, 0x00, 0x00, 0x5b, 0xe8, 0x01, 0x00, 0x00, 0x00, 0xcc, 0x5a,
        ],
    );
    steps(&mut cpu, &mem, 4);
    assert_eq!(cpu.regs[RBX], 0x1234);
    assert_eq!(cpu.regs[RDX], CODE + 11);
    assert_eq!(cpu.regs[RSP], STACK_TOP);
}

#[test]
fn division_by_zero_faults_without_changing_state() {
    let (mut cpu, mem) = machine(
        "xor edx, edx; xor ecx, ecx; div ecx",
        &[0x31, 0xd2, 0x31, 0xc9, 0xf7, 0xf1],
    );
    cpu.regs[RAX] = 7;
    steps(&mut cpu, &mem, 2);
    let before = cpu.clone();
    assert_eq!(
        step(&mut cpu, &mem),
        Exit::Fault {
            rip: CODE + 4,
            kind: CpuFault::DivideError
        }
    );
    assert_eq!(cpu, before);
}

#[test]
fn rep_movsb_copies_rcx_bytes() {
    let (mut cpu, mem) = machine(
        "mov ecx, 5; rep movsb",
        &[0xb9, 0x05, 0x00, 0x00, 0x00, 0xf3, 0xa4],
    );
    mem.write(DATA, b"hello world").unwrap();
    cpu.regs[RSI] = DATA;
    cpu.regs[RDI] = DATA + 0x100;
    steps(&mut cpu, &mem, 2);
    let mut out = [0u8; 6];
    mem.read(DATA + 0x100, &mut out).unwrap();
    assert_eq!(&out, b"hello\0");
    assert_eq!((cpu.regs[RCX], cpu.regs[RSI]), (0, DATA + 5));
}

#[test]
fn jumping_into_a_trap_region_exits_with_its_address() {
    let (mut cpu, mem) = machine(
        "mov rax, 0x10000000; jmp rax",
        &[0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x10, 0xff, 0xe0],
    );
    assert_eq!(run(&mut cpu, &mem, 10), Exit::Trap(TRAP));
}

#[test]
fn syscall_exits_past_the_instruction() {
    let (mut cpu, mem) = machine(
        "mov eax, 60; syscall",
        &[0xb8, 0x3c, 0x00, 0x00, 0x00, 0x0f, 0x05],
    );
    assert_eq!(run(&mut cpu, &mem, 10), Exit::Syscall);
    assert_eq!((cpu.regs[RAX], cpu.rip), (60, CODE + 7));
}

#[test]
fn mul_and_imul_report_overflow() {
    let (mut cpu, mem) = machine(
        "mov rax, -1; mov rcx, 2; mul rcx",
        &[
            0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, 0x48, 0xc7, 0xc1, 0x02, 0x00, 0x00, 0x00,
            0x48, 0xf7, 0xe1,
        ],
    );
    steps(&mut cpu, &mem, 3);
    assert_eq!((cpu.regs[RAX], cpu.regs[RDX]), (u64::MAX - 1, 1));
    assert!(cpu.flag(CF) && cpu.flag(OF));

    let (mut cpu, mem) = machine(
        "mov eax, 0x40000000; imul eax, eax, 4",
        &[0xb8, 0x00, 0x00, 0x00, 0x40, 0x6b, 0xc0, 0x04],
    );
    steps(&mut cpu, &mem, 2);
    assert_eq!(cpu.regs[RAX], 0);
    assert!(cpu.flag(OF));
}

#[test]
fn writes_to_read_only_code_fault_at_the_instruction() {
    let (mut cpu, mem) = machine(
        "mov dword ptr [rip], 0",
        &[0xc7, 0x05, 0, 0, 0, 0, 0, 0, 0, 0],
    );
    match step(&mut cpu, &mem) {
        Exit::Fault {
            rip,
            kind: CpuFault::Memory(f),
        } => {
            assert_eq!((rip, f.addr, f.access), (CODE, CODE + 10, Access::Write));
        }
        other => panic!("expected a memory fault, got {other:?}"),
    }
    assert_eq!(cpu.rip, CODE);
}

#[test]
fn conditional_moves_sets_and_extensions() {
    let (mut cpu, mem) = machine(
        "mov ecx, 3; cmp ecx, 3; sete al; cmovne ecx, eax; movsx edx, al; movzx esi, cl",
        &[
            0xb9, 0x03, 0x00, 0x00, 0x00, 0x83, 0xf9, 0x03, 0x0f, 0x94, 0xc0, 0x0f, 0x45, 0xc8,
            0x0f, 0xbe, 0xd0, 0x0f, 0xb6, 0xf1,
        ],
    );
    cpu.regs[RAX] = 0xff00;
    steps(&mut cpu, &mem, 6);
    assert_eq!(cpu.regs[RAX] & 0xff, 1);
    assert_eq!(cpu.regs[RCX], 3);
    assert_eq!((cpu.regs[RDX], cpu.regs[RSI]), (1, 3));
}

#[test]
fn bit_scans_and_tests() {
    let (mut cpu, mem) = machine(
        "mov eax, 8; tzcnt edx, eax; bsr esi, eax; bt eax, 3",
        &[
            0xb8, 0x08, 0x00, 0x00, 0x00, 0xf3, 0x0f, 0xbc, 0xd0, 0x0f, 0xbd, 0xf0, 0x0f, 0xba,
            0xe0, 0x03,
        ],
    );
    steps(&mut cpu, &mem, 4);
    assert_eq!((cpu.regs[RDX], cpu.regs[RSI]), (3, 3));
    assert!(cpu.flag(CF));
}

#[test]
fn shifts_and_rotates() {
    let (mut cpu, mem) = machine(
        "sar eax, 1; shl rax, 63; ror eax, 4",
        &[0xd1, 0xf8, 0x48, 0xc1, 0xe0, 0x3f, 0xc1, 0xc8, 0x04],
    );
    cpu.regs[RAX] = 0x8000_0003;
    steps(&mut cpu, &mem, 1);
    assert_eq!(cpu.regs[RAX], 0xc000_0001);
    assert!(cpu.flag(CF));
    steps(&mut cpu, &mem, 1);
    assert_eq!(cpu.regs[RAX], 1 << 63);
    steps(&mut cpu, &mem, 1);
    assert_eq!(cpu.regs[RAX], 0);
}

#[test]
fn unimplemented_instructions_are_reported() {
    let (mut cpu, mem) = machine("fbld tbyte ptr [rax]", &[0xdf, 0x20]);
    assert_eq!(
        step(&mut cpu, &mem),
        Exit::Fault {
            rip: CODE,
            kind: CpuFault::Unimplemented {
                map: Map::Primary,
                op: 0xdf
            }
        }
    );
}

#[test]
fn address_size_prefix_makes_string_instructions_use_esi() {
    let (mut cpu, mem) = machine("lodsb byte ptr [esi]", &[0x67, 0xac]);
    mem.write(DATA, &[0x5a]).unwrap();
    cpu.regs[RSI] = 0xffff_ffff_0000_0000 | DATA;
    steps(&mut cpu, &mem, 1);
    assert_eq!((cpu.regs[RAX] & 0xff, cpu.regs[RSI]), (0x5a, DATA + 1));
}

#[test]
fn zero_count_double_shift_still_zero_extends_its_register() {
    let (mut cpu, mem) = machine("shld eax, ecx, 0", &[0x0f, 0xa4, 0xc8, 0x00]);
    cpu.regs[RAX] = u64::MAX;
    steps(&mut cpu, &mem, 1);
    assert_eq!(cpu.regs[RAX], 0xffff_ffff);
}

#[test]
fn cpuid_reports_a_zen2_with_lzcnt_and_avx2() {
    let (mut cpu, mem) = machine("cpuid", &[0x0f, 0xa2]);
    cpu.regs[RAX] = 0x8000_0001;
    steps(&mut cpu, &mem, 1);
    assert_ne!(cpu.regs[RCX] & 1 << 5, 0);
    cpu.rip = CODE;
    cpu.regs[RAX] = 7;
    cpu.regs[RCX] = 0;
    steps(&mut cpu, &mem, 1);
    assert_ne!(cpu.regs[RBX] & 1 << 5, 0);
}

#[test]
fn locked_increments_from_several_threads_are_atomic() {
    let (cpu, mem) = machine(
        "l: lock inc qword ptr [rdi]; dec ecx; jnz l; hlt",
        &[0xf0, 0x48, 0xff, 0x07, 0xff, 0xc9, 0x75, 0xf8, 0xf4],
    );
    let threads = 4;
    let per_thread = 20_000;
    std::thread::scope(|s| {
        for _ in 0..threads {
            let mut cpu = cpu.clone();
            let mem = mem.clone();
            s.spawn(move || {
                cpu.regs[RDI] = DATA;
                cpu.regs[RCX] = per_thread;
                let exit = run(&mut cpu, &mem, u64::MAX);
                assert!(matches!(
                    exit,
                    Exit::Fault {
                        kind: CpuFault::Privileged,
                        ..
                    }
                ));
            });
        }
    });
    assert_eq!(mem.read_u64(DATA).unwrap(), threads * per_thread);
}

fn ps(v: [f32; 8]) -> [u64; 4] {
    core::array::from_fn(|i| v[2 * i].to_bits() as u64 | (v[2 * i + 1].to_bits() as u64) << 32)
}

#[test]
fn vex_256_operations_cover_both_halves_without_touching_the_sources() {
    let (mut cpu, mem) = machine("vaddps ymm1, ymm2, ymm3", &[0xc5, 0xec, 0x58, 0xcb]);
    cpu.xmm[2] = ps([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    cpu.xmm[3] = ps([10.0; 8]);
    steps(&mut cpu, &mem, 1);
    assert_eq!(
        cpu.xmm[1],
        ps([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    );
    assert_eq!(cpu.xmm[2], ps([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
}

#[test]
fn vex_128_clears_the_upper_half_and_legacy_sse_keeps_it() {
    let (mut cpu, mem) = machine(
        "vpxor xmm4, xmm4, xmm4; pxor xmm5, xmm5",
        &[0xc5, 0xd9, 0xef, 0xe4, 0x66, 0x0f, 0xef, 0xed],
    );
    cpu.xmm[4] = [1, 2, 3, 4];
    cpu.xmm[5] = [1, 2, 3, 4];
    steps(&mut cpu, &mem, 2);
    assert_eq!(cpu.xmm[4], [0; 4]);
    assert_eq!(cpu.xmm[5], [0, 0, 3, 4]);
}

#[test]
fn fused_multiply_add_rounds_once() {
    let (mut cpu, mem) = machine(
        "vfmadd231ps xmm0, xmm1, xmm2",
        &[0xc4, 0xe2, 0x71, 0xb8, 0xc2],
    );
    let a = 1.0f32 + f32::EPSILON;
    let c = -(1.0f32 + 2.0 * f32::EPSILON);
    cpu.xmm[1] = [a.to_bits() as u64, 0, 0, 0];
    cpu.xmm[2] = [a.to_bits() as u64, 0, 0, 0];
    cpu.xmm[0] = [c.to_bits() as u64, 0, 0, 0];
    steps(&mut cpu, &mem, 1);
    let r = f32::from_bits(cpu.xmm[0][0] as u32);
    assert_eq!(r, f32::EPSILON * f32::EPSILON);
}

#[test]
fn broadcast_fills_every_lane() {
    let (mut cpu, mem) = machine("vpbroadcastd ymm3, xmm4", &[0xc4, 0xe2, 0x7d, 0x58, 0xdc]);
    cpu.xmm[4] = [0x1234_5678, 0, 0, 0];
    steps(&mut cpu, &mem, 1);
    assert_eq!(cpu.xmm[3], [0x1234_5678_1234_5678; 4]);
}

#[test]
fn pdep_scatters_low_bits_into_the_mask() {
    let (mut cpu, mem) = machine("pdep rax, rbx, rcx", &[0xc4, 0xe2, 0xe3, 0xf5, 0xc1]);
    cpu.regs[RBX] = 0b1011;
    cpu.regs[RCX] = 0b1111_0000_0000;
    steps(&mut cpu, &mem, 1);
    assert_eq!(cpu.regs[RAX], 0b1011_0000_0000);
}

#[test]
fn blendv_takes_its_mask_from_any_register_named_by_the_immediate() {
    let (mut cpu, mem) = machine(
        "vblendvps xmm0, xmm1, xmm2, xmm9",
        &[0xc4, 0xe3, 0x71, 0x4a, 0xc2, 0x90],
    );
    cpu.xmm[1] = [0x1111_1111_1111_1111; 4];
    cpu.xmm[2] = [0x2222_2222_2222_2222; 4];
    cpu.xmm[9] = [0x8000_0000_0000_0000, 0x0000_0000_8000_0000, 0, 0];
    steps(&mut cpu, &mem, 1);
    assert_eq!(
        cpu.xmm[0],
        [0x2222_2222_1111_1111, 0x1111_1111_2222_2222, 0, 0]
    );
}

#[test]
fn gather_loads_masked_elements_and_clears_the_mask() {
    let (mut cpu, mem) = machine(
        "vpgatherdd xmm0, dword ptr [rdi + xmm4*4], xmm5",
        &[0xc4, 0xe2, 0x51, 0x90, 0x04, 0xa7],
    );
    for k in 0..8u32 {
        mem.write(DATA + 4 * k as u64, &(100 + k).to_le_bytes())
            .unwrap();
    }
    cpu.regs[RDI] = DATA;
    cpu.xmm[4] = [3 | 1 << 32, 7, 0, 0];
    cpu.xmm[5] = [0x8000_0000_8000_0000, 0x8000_0000, 0, 0];
    cpu.xmm[0] = [0, 0xdead_beef << 32, 0, 0];
    steps(&mut cpu, &mem, 1);
    assert_eq!(cpu.xmm[0][0], 103 | 101 << 32);
    assert_eq!(cpu.xmm[0][1], 107 | 0xdead_beef << 32);
    assert_eq!(cpu.xmm[5], [0; 4]);
}

#[test]
fn x87_loads_adds_and_stores_extended_values() {
    let (mut cpu, mem) = machine(
        "fld tbyte ptr [rdi]; fld st(0); faddp st(1), st; fstp tbyte ptr [rdi+16]",
        &[0xdb, 0x2f, 0xd9, 0xc0, 0xde, 0xc1, 0xdb, 0x7f, 0x10],
    );
    mem.write(DATA, &0xc000_0000_0000_0000u64.to_le_bytes())
        .unwrap();
    mem.write(DATA + 8, &0x3fffu16.to_le_bytes()).unwrap();
    cpu.regs[RDI] = DATA;
    steps(&mut cpu, &mem, 4);
    let mut out = [0u8; 10];
    mem.read(DATA + 16, &mut out).unwrap();
    assert_eq!(
        u64::from_le_bytes(out[..8].try_into().unwrap()),
        0xc000_0000_0000_0000
    );
    assert_eq!(u16::from_le_bytes([out[8], out[9]]), 0x4000);
    assert_eq!((cpu.ftag, (cpu.fsw >> 11) & 7), (0, 0));
}

#[test]
fn x87_integer_division_rounds_and_fcomip_sets_flags() {
    let (mut cpu, mem) = machine(
        "fild dword ptr [rdi]; fild dword ptr [rdi+4]; fdivp st(1), st; \
         fistp qword ptr [rdi+8]; fld1; fldz; fcomip st, st(1); fstp st(0)",
        &[
            0xdb, 0x07, 0xdb, 0x47, 0x04, 0xde, 0xf9, 0xdf, 0x7f, 0x08, 0xd9, 0xe8, 0xd9, 0xee,
            0xdf, 0xf1, 0xdd, 0xd8,
        ],
    );
    mem.write(DATA, &7i32.to_le_bytes()).unwrap();
    mem.write(DATA + 4, &2i32.to_le_bytes()).unwrap();
    cpu.regs[RDI] = DATA;
    steps(&mut cpu, &mem, 8);
    assert_eq!(mem.read_u64(DATA + 8).unwrap(), 4);
    assert!(cpu.flag(CF) && !cpu.flag(ZF) && !cpu.flag(PF));
    assert_eq!(cpu.ftag, 0);
}

#[test]
fn extended_values_round_to_the_nearest_double() {
    use super::x87::{from_extended, to_extended};
    for v in [1.5, -0.0, f64::INFINITY, 5e-324, -1e300, 0.1] {
        let (m, se) = to_extended(v);
        assert_eq!(from_extended(m, se).to_bits(), v.to_bits(), "{v}");
    }
    assert!(from_extended(0xc000_0000_0000_0000, 0x7fff).is_nan());
    let just_above_half = 1 << 63 | 1 << 10 | 1;
    assert_eq!(from_extended(just_above_half, 0x3fff), 1.0 + f64::EPSILON);
    assert_eq!(from_extended(1 << 63 | 1 << 10, 0x3fff), 1.0);
    assert_eq!(from_extended(1 << 63, 0x3fff + 1024), f64::INFINITY);
}
