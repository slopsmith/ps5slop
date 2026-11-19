//! libkernel. Which of these names libkernel really exports, and with which
//! error conventions, is unverified (see `docs/hle-kernel.md`).

use crate::{CallError, Env, Function, Outcome};

pub static FUNCTIONS: &[Function] = &[
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
    let sink: &mut dyn std::io::Write = match fd {
        1 => &mut *env.stdout,
        2 => &mut *env.stderr,
        _ => return Ok(Outcome::Return(-1i64 as u64)),
    };
    match sink.write_all(&data).and_then(|_| sink.flush()) {
        Ok(()) => Ok(Outcome::Return(len)),
        Err(_) => Ok(Outcome::Return(-1i64 as u64)),
    }
}

/// `exit(status)`.
fn exit(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Exit(env.arg(0) as i32))
}
