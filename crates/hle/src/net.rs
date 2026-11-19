//! libSceNet: set-up succeeds, and the console is offline.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{CallError, Env, Function, Outcome, Variable};

const LOOPBACK: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

pub static VARIABLES: &[Variable] = &[
    Variable {
        library: "libSceNet",
        name: "in6addr_any",
        init: &[0; 16],
    },
    Variable {
        library: "libSceNet",
        name: "in6addr_loopback",
        init: &LOOPBACK,
    },
];

pub static FUNCTIONS: &[Function] = &[
    f("sceNetInit", ok),
    f("sceNetTerm", ok),
    f("sceNetPoolCreate", pool_create),
    f("sceNetPoolDestroy", ok),
    f("sceNetGetMacAddress", mac_address),
    f("sceNetEtherNtostr", ether_ntostr),
];

/// A locally administered address (our choice).
const MAC: [u8; 6] = [0x02, 0x50, 0x53, 0x35, 0x00, 0x01];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceNet",
        name,
        func,
    }
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

/// `sceNetPoolCreate(name, size, flags)`: a new pool id.
fn pool_create(_: &mut Env) -> Result<Outcome, CallError> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    Ok(Outcome::Return(NEXT.fetch_add(1, Ordering::Relaxed)))
}

/// `sceNetGetMacAddress(addr, flags)`.
fn mac_address(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(0), &MAC)?;
    Ok(Outcome::Return(0))
}

/// `sceNetEtherNtostr(addr, str, len)`: `xx:xx:xx:xx:xx:xx`.
fn ether_ntostr(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr, out, len) = (env.arg(0), env.arg(1), env.arg(2) as usize);
    let mut mac = [0u8; 6];
    env.mem.read(addr, &mut mac)?;
    let mut text = mac.map(|b| format!("{b:02x}")).join(":").into_bytes();
    text.push(0);
    if len < text.len() {
        return Ok(Outcome::Return(crate::dmem::sce_error(
            crate::errno::EINVAL,
        )));
    }
    env.mem.write(out, &text)?;
    Ok(Outcome::Return(0))
}
