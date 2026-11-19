//! Command-buffer builders (`sceAgcDcb*`): each appends PM4 packets and
//! returns where they went, or null when the buffer is full. Ported from
//! SharpEmu's `AgcExports.CommandPackets.cs` and checked against console
//! measurements (`oracle.rs`).

use super::cmdbuf::{IT_NOP, emit, pm4};
use super::ret;
use crate::{CallError, Env, Function, Outcome};

pub(super) static FUNCTIONS: &[Function] = &[f("sceAgcDcbResetQueue", dcb_reset_queue)];

pub(super) static STUBS: &[Function] = &[];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
        name,
        func,
    }
}

/// The shader-type/index field SharpEmu gives the reset marker.
const R_DRAW_RESET: u32 = 0x05;

/// `sceAgcDcbResetQueue(buffer, queue, mode)`: a NOP marking the reset,
/// shorter than the console's mode-dependent sequence (see `oracle.rs`).
fn dcb_reset_queue(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(env, buffer, &[pm4(2, IT_NOP, R_DRAW_RESET), 0])?)
}
