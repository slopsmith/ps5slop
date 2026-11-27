//! A guest `Env` for host function tests.

use memory::{Memory, PAGE_SIZE, Prot};

use crate::{Cpu, Env, Hle, Kernel, Outcome, Thread, TlsTemplate, Vfs};

/// Runs `test` with an `Env` over 2 MiB of guest memory at `area`, whose
/// second page holds the thread's TCB.
pub fn with_env(area: u64, test: impl FnOnce(&mut Env, &Memory)) {
    let mem = Memory::new(1 << 40).unwrap();
    mem.map(area, 0x20_0000, Prot::RW, "test").unwrap();
    let hle = Hle::new();
    hle.map(&mem).unwrap();
    let kernel = Kernel::new(
        hle,
        mem,
        TlsTemplate::default(),
        Box::new(std::io::sink()),
        Box::new(std::io::sink()),
        Vfs::default(),
    );
    let mem = kernel.memory();
    let mut cpu = Cpu::default();
    let mut env = Env {
        cpu: &mut cpu,
        mem: &mem,
        kernel: &kernel,
        thread: Thread {
            tcb: area + PAGE_SIZE,
        },
    };
    test(&mut env, &mem);
}

/// Calls `f` with integer arguments and returns its result.
pub fn call_for(env: &mut Env, f: crate::HostFn, args: &[u64]) -> u64 {
    for (k, &a) in args.iter().enumerate() {
        env.cpu.regs[[7, 6, 2, 1, 8, 9][k]] = a;
    }
    match f(env) {
        Ok(Outcome::Return(r)) => r,
        _ => panic!("host function did not return"),
    }
}

/// Calls `f` with integer arguments and checks that it returns 0.
pub fn call(env: &mut Env, f: crate::HostFn, args: &[u64]) {
    assert_eq!(call_for(env, f, args), 0);
}
