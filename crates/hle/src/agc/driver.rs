//! libSceAgcDriver's resource registration: owners and named GPU resources,
//! kept for debugging tools on a console. We hand out ids and keep the
//! names. Ported from SharpEmu's `AgcExports.DriverResources.cs`.

use std::collections::HashMap;

use super::{ERROR_INVALID_ARGUMENT, ret};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

const NAME_MAX: usize = 256;
/// The owner `sceAgcDriverGetDefaultOwner` reports.
const DEFAULT_OWNER: u32 = 1;
/// Bytes of registration memory per resource and per owner (SharpEmu's).
const BYTES_PER_RESOURCE: u64 = 0x118;
const BYTES_PER_OWNER: u64 = 0x1E0;

pub(super) struct Registry {
    owners: HashMap<u32, String>,
    /// Resources by handle: owner, name.
    resources: HashMap<u32, (u32, String)>,
    next_owner: u32,
    next_resource: u32,
    default_owner: u32,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            owners: HashMap::new(),
            resources: HashMap::new(),
            next_owner: 1,
            next_resource: 1,
            default_owner: DEFAULT_OWNER,
        }
    }
}

pub(super) static FUNCTIONS: &[Function] = &[
    f(
        "sceAgcDriverGetResourceRegistrationMaxNameLength",
        max_name_length,
    ),
    f(
        "sceAgcDriverQueryResourceRegistrationUserMemoryRequirements",
        memory_requirements,
    ),
    f("sceAgcDriverInitResourceRegistration", init_registration),
    f("sceAgcDriverGetDefaultOwner", default_owner),
    f("sceAgcDriverRegisterDefaultOwner", register_default_owner),
    f("sceAgcDriverRegisterOwner", register_owner),
    f("sceAgcDriverRegisterResource", register_resource),
    f("sceAgcDriverUnregisterResource", unregister_resource),
    f(
        "sceAgcDriverUnregisterAllResourcesForOwner",
        unregister_owner_resources,
    ),
    f("sceAgcDriverUnregisterOwnerAndResources", unregister_owner),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub(super) static STUBS: &[Function] = &[
    f("sceAgcDriverSetTFRing", ok),
    f("sceAgcDriverSetHsOffchipParam", ok),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgcDriver",
        name,
        func,
    }
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn max_name_length(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(0) == 0 {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    env.mem
        .write(env.arg(0), &(NAME_MAX as u32).to_le_bytes())?;
    ret(0)
}

/// `(&size, resources, owners)`.
fn memory_requirements(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, resources, owners) = (env.arg(0), env.arg(1), env.arg(2));
    let size = resources
        .checked_mul(BYTES_PER_RESOURCE)
        .and_then(|r| r.checked_add(owners.checked_mul(BYTES_PER_OWNER)?));
    let (Some(size), true) = (size, out != 0 && resources != 0 && owners != 0) else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    env.mem.write_u64(out, size)?;
    ret(0)
}

/// `(memory, size, owners)`: the memory is not used.
fn init_registration(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(0) == 0 || env.arg(1) == 0 || env.arg(2) == 0 {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    *lock(&env.kernel.agc.registry) = Registry::default();
    ret(0)
}

fn default_owner(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(0) == 0 {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    env.mem.write(env.arg(0), &DEFAULT_OWNER.to_le_bytes())?;
    ret(0)
}

fn register_default_owner(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.agc.registry).default_owner = env.arg(0) as u32;
    ret(0)
}

/// `(&owner, name)`: a fresh owner id other than the default owner.
fn register_owner(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, name_at) = (env.arg(0), env.arg(1));
    if out == 0 || name_at == 0 {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    let name = String::from_utf8_lossy(&env.mem.read_cstr(name_at, NAME_MAX)?).into_owned();
    let mut reg = lock(&env.kernel.agc.registry);
    let mut owner = reg.next_owner;
    while owner == reg.default_owner || reg.owners.contains_key(&owner) {
        owner += 1;
    }
    reg.next_owner = owner + 1;
    reg.owners.insert(owner, name);
    drop(reg);
    env.mem.write(out, &owner.to_le_bytes())?;
    ret(0)
}

/// `(address, owner, name, size, type, flags)`.
fn register_resource(env: &mut Env) -> Result<Outcome, CallError> {
    let (owner, name_at) = (env.arg(1) as u32, env.arg(2));
    let name = match name_at {
        0 => String::new(),
        at => String::from_utf8_lossy(&env.mem.read_cstr(at, NAME_MAX)?).into_owned(),
    };
    let mut reg = lock(&env.kernel.agc.registry);
    let handle = reg.next_resource;
    reg.next_resource = handle + 1;
    reg.resources.insert(handle, (owner, name));
    ret(0)
}

fn unregister_resource(env: &mut Env) -> Result<Outcome, CallError> {
    let removed = lock(&env.kernel.agc.registry)
        .resources
        .remove(&(env.arg(0) as u32));
    ret(if removed.is_some() {
        0
    } else {
        ERROR_INVALID_ARGUMENT
    })
}

fn unregister_owner_resources(env: &mut Env) -> Result<Outcome, CallError> {
    let owner = env.arg(0) as u32;
    lock(&env.kernel.agc.registry)
        .resources
        .retain(|_, r| r.0 != owner);
    ret(0)
}

fn unregister_owner(env: &mut Env) -> Result<Outcome, CallError> {
    let owner = env.arg(0) as u32;
    let mut reg = lock(&env.kernel.agc.registry);
    if reg.owners.remove(&owner).is_none() {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    reg.resources.retain(|_, r| r.0 != owner);
    ret(0)
}
