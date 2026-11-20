//! libSceSysmodule: loading optional system modules by id. The libraries a
//! program can call are all provided by the HLE layer, so loading one
//! succeeds without doing anything. Which ids name which modules is not
//! recorded yet.

use crate::{CallError, Env, Function, Outcome};

pub static FUNCTIONS: &[Function] = &[
    Function {
        library: "libSceSysmodule",
        name: "sceSysmoduleLoadModuleInternal",
        func: load_module,
    },
    Function {
        library: "libSceSysmodule",
        name: "sceSysmoduleLoadModule",
        func: load_module,
    },
    Function {
        library: "libSceSysmodule",
        name: "sceSysmoduleUnloadModule",
        func: load_module,
    },
    Function {
        library: "libSceSysmodule",
        name: "sceSysmoduleIsLoaded",
        func: load_module,
    },
    Function {
        library: "libSceSysmodule",
        name: "sceSysmoduleGetModuleInfoForUnwind",
        func: crate::rtld::module_info_for_unwind,
    },
];

/// `sceSysmoduleLoadModule(id)`, its `Internal` variant, `Unload` and
/// `IsLoaded`: returns 0 (success, or loaded).
fn load_module(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}
