//! libkernel. Which of these names libkernel really exports, and with which
//! error conventions, is unverified (see `docs/hle-kernel.md`).

use memory::{PAGE_SIZE, Prot};

use crate::{CallError, Env, Function, Outcome, Variable};

pub static VARIABLES: &[Variable] = &[Variable {
    library: "libkernel",
    name: "__stack_chk_guard",
    init: &0x2f8a_1c3e_5d7b_9e00u64.to_le_bytes(),
}];

pub static FUNCTIONS: &[Function] = &[
    Function {
        library: "libkernel",
        name: "__stack_chk_fail",
        func: stack_chk_fail,
    },
    Function {
        library: "libkernel",
        name: "mmap",
        func: mmap,
    },
    Function {
        library: "libkernel",
        name: "munmap",
        func: munmap,
    },
    Function {
        library: "libkernel",
        name: "write",
        func: write,
    },
    Function {
        library: "libkernel",
        name: "exit",
        func: exit,
    },
    Function {
        library: crate::INTERNAL,
        name: crate::PROGRAM_EXIT,
        func: exit,
    },
];

/// Guest writes larger than this are cut, like a short write.
const MAX_WRITE: u64 = 1 << 20;

/// `write(fd, buf, len)`: fd 1 and 2 go to our stdout and stderr; other
/// descriptors do not exist yet and fail with -1.
fn write(env: &mut Env) -> Result<Outcome, CallError> {
    let (fd, buf, len) = (env.arg(0) as i32, env.arg(1), env.arg(2).min(MAX_WRITE));
    let mut data = vec![0u8; len as usize];
    env.mem.read(buf, &mut data)?;
    match env.kernel.write_output(fd, &data) {
        Ok(()) => Ok(Outcome::Return(len)),
        Err(_) => Ok(Outcome::Return(-1i64 as u64)),
    }
}

/// `exit(status)`.
fn exit(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Exit(env.arg(0) as i32))
}

/// `__stack_chk_fail()`: the guest found its stack canary overwritten.
fn stack_chk_fail(_: &mut Env) -> Result<Outcome, CallError> {
    Err(CallError::Abort("stack smashing detected"))
}

/// Where `mmap` puts mappings without a fixed address: the PS4 user area
/// (shadPS4's `address_space.cpp`; unverified for PS5).
const USER_AREA: u64 = 0x10_0000_0000;

const PROT_READ: u64 = 1;
const PROT_WRITE: u64 = 2;
const PROT_EXEC: u64 = 4;
/// FreeBSD's `MAP_FIXED` and `MAP_ANON`.
const MAP_FIXED: u64 = 0x10;
const MAP_ANON: u64 = 0x1000;
const MAP_FAILED: u64 = u64::MAX;

/// `mmap(addr, len, prot, flags, fd, offset)` for anonymous memory only.
/// Failures return `MAP_FAILED` without setting errno yet.
fn mmap(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr, len, prot, flags) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    if flags & MAP_ANON == 0 || len == 0 {
        return Ok(Outcome::Return(MAP_FAILED));
    }
    let len = len.next_multiple_of(PAGE_SIZE);
    let prot = Prot::new(
        prot & PROT_READ != 0,
        prot & PROT_WRITE != 0,
        prot & PROT_EXEC != 0,
    );
    let at = if flags & MAP_FIXED != 0 {
        if !addr.is_multiple_of(PAGE_SIZE) {
            return Ok(Outcome::Return(MAP_FAILED));
        }
        addr
    } else {
        match env.mem.find_free(addr.max(USER_AREA), len) {
            Some(a) => a,
            None => return Ok(Outcome::Return(MAP_FAILED)),
        }
    };
    match env.mem.map(at, len, prot, "mmap") {
        Ok(()) => Ok(Outcome::Return(at)),
        Err(_) => Ok(Outcome::Return(MAP_FAILED)),
    }
}

/// `munmap(addr, len)`: only whole mappings made by `mmap`.
fn munmap(env: &mut Env) -> Result<Outcome, CallError> {
    match env.mem.unmap(env.arg(0)) {
        Ok(()) => Ok(Outcome::Return(0)),
        Err(_) => Ok(Outcome::Return(-1i64 as u64)),
    }
}
