//! libkernel. Which of these names libkernel really exports, and with which
//! error conventions, is unverified (see `docs/hle-kernel.md`).

use memory::{PAGE_SIZE, Prot};

use crate::{CallError, Env, Function, Outcome, Variable, errno};

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
        name: "__error",
        func: error,
    },
    Function {
        library: "libkernel",
        name: "ioctl",
        func: ioctl,
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
        func: crate::file::write,
    },
    Function {
        library: "libkernel",
        name: "_write",
        func: crate::file::write,
    },
    Function {
        library: "libkernel",
        name: "exit",
        func: exit,
    },
    Function {
        library: crate::INTERNAL,
        name: crate::LOADER_TEARDOWN,
        func: loader_teardown,
    },
    Function {
        library: crate::INTERNAL,
        name: crate::CALLBACK_RETURN,
        func: stray_callback_return,
    },
];

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

/// `__error()`: the address of the calling thread's errno.
fn error(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(env.thread.errno_addr()))
}

/// `ioctl(fd, request, ...)`: no descriptor supports any request yet.
fn ioctl(env: &mut Env) -> Result<Outcome, CallError> {
    env.fail(errno::ENOTTY)
}

/// Reached only when guest code returns to the callback stub outside a
/// callback, which a correct program never does.
fn stray_callback_return(_: &mut Env) -> Result<Outcome, CallError> {
    Err(CallError::Abort(
        "return to the callback stub outside a callback",
    ))
}

/// The entry point's teardown callback: nothing to tear down yet.
fn loader_teardown(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}
