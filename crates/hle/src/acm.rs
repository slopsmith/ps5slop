//! libSceAcm placeholders. FMOD creates a context while it starts and gives
//! up on all audio when that fails; it has not been seen starting a batch.

use crate::{CallError, Env, Function, Outcome};

pub static STUBS: &[Function] = &[
    f("sceAcmContextCreate", ok),
    f("sceAcmContextDestroy", ok),
    f("sceAcmBatchStartBuffers", ok),
    f("sceAcmBatchWait", ok),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAcm",
        name,
        func,
    }
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

