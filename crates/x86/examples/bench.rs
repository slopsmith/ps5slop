use memory::{Memory, PAGE_SIZE, Prot};
use x86::{Cpu, Exit};
fn main() {
    // loop: mov rax,[rdi]; add rax,1; mov [rdi],rax; dec ecx; jnz loop; hlt
    let code = [
        0x48, 0x8b, 0x07, 0x48, 0x83, 0xc0, 0x01, 0x48, 0x89, 0x07, 0xff, 0xc9, 0x75, 0xf2, 0xf4,
    ];
    let mem = Memory::new(1 << 36).unwrap();
    mem.map(0x10000, PAGE_SIZE, Prot::RX, "code").unwrap();
    mem.write_host(0x10000, &code).unwrap();
    mem.map(0x20000, PAGE_SIZE, Prot::RW, "data").unwrap();
    let mut cpu = Cpu {
        rip: 0x10000,
        ..Cpu::default()
    };
    cpu.regs[7] = 0x20000;
    let n = 10_000_000u64;
    cpu.regs[1] = n;
    let t = std::time::Instant::now();
    let e = x86::run(&mut cpu, &mem, u64::MAX);
    let s = t.elapsed().as_secs_f64();
    assert!(matches!(e, Exit::Fault { .. }));
    assert_eq!(mem.read_u64(0x20000).unwrap(), n);
    println!(
        "{:.1} M instructions/s ({:.2} s)",
        (5 * n) as f64 / s / 1e6,
        s
    );
}
