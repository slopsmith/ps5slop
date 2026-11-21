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
    Function {
        library: "libkernel",
        name: "inet_ntop",
        func: inet_ntop,
    },
    Function {
        library: "libkernel",
        name: "inet_pton",
        func: inet_pton,
    },
    Function {
        library: "libSceNetCtl",
        name: "sceNetCtlGetInfo",
        func: netctl_info,
    },
    f("sceNetInit", ok),
    f("sceNetTerm", ok),
    f("sceNetPoolCreate", pool_create),
    f("sceNetPoolDestroy", ok),
    f("sceNetGetMacAddress", mac_address),
    f("sceNetEtherNtostr", ether_ntostr),
    f("sceNetInetPton", inet_pton),
    f("sceNetInetNtop", inet_ntop),
    f("sceNetSocket", socket),
    f("sceNetErrnoLoc", errno_loc),
];

/// libSceNet's errors: `0x80410100` plus the errno (shadPS4, PS4;
/// unverified for the PS5).
const fn net_error(errno: i32) -> u64 {
    (0x8041_0100u32 + errno as u32) as u64
}

/// A locally administered address (our choice).
const MAC: [u8; 6] = [0x02, 0x50, 0x53, 0x35, 0x00, 0x01];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceNet",
        name,
        func,
    }
}

/// `sceNetSocket(name, family, type, protocol)`: the network is down.
fn socket(env: &mut Env) -> Result<Outcome, CallError> {
    env.set_errno(crate::errno::ENETDOWN)?;
    Ok(Outcome::Return(net_error(crate::errno::ENETDOWN)))
}

/// `sceNetErrnoLoc()`: the calling thread's errno, shared with libkernel's.
fn errno_loc(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(env.thread.errno_addr()))
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

/// `sceNetPoolCreate(name, size, flags)`: a new pool id.
fn pool_create(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(NEXT_POOL.fetch_add(1, Ordering::Relaxed)))
}

static NEXT_POOL: AtomicU64 = AtomicU64::new(1);

pub(crate) fn save(e: &mut savestate::Encoder) {
    e.put(&NEXT_POOL.load(Ordering::Relaxed));
}

pub(crate) fn load(d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
    NEXT_POOL.store(d.get()?, Ordering::Relaxed);
    Ok(())
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

/// `sceNetCtlGetInfo(code, info)` for a disconnected interface, using
/// OpenOrbis's info codes: our MAC, MTU 1500, link down, zero addresses.
fn netctl_info(env: &mut Env) -> Result<Outcome, CallError> {
    let mut info = [0u8; 256];
    match env.arg(0) as u32 {
        2 => info[..6].copy_from_slice(&MAC),
        3 => info[..4].copy_from_slice(&1500u32.to_le_bytes()),
        14..=18 => info[..8].copy_from_slice(b"0.0.0.0\0"),
        1 | 4..=13 | 19..=21 => {}
        _ => {
            return Ok(Outcome::Return(crate::dmem::sce_error(
                crate::errno::EINVAL,
            )));
        }
    }
    env.mem.write(env.arg(1), &info)?;
    Ok(Outcome::Return(0))
}

/// FreeBSD's `AF_INET` and `AF_INET6`.
const AF_INET: u64 = 2;
const AF_INET6: u64 = 28;

/// `inet_pton(family, text, out)`: 1 when parsed, 0 for bad text.
fn inet_pton(env: &mut Env) -> Result<Outcome, CallError> {
    let (family, out) = (env.arg(0) as u32 as u64, env.arg(2));
    let text = crate::libc::cstr(env, env.arg(1))?;
    let text = String::from_utf8_lossy(&text);
    let bytes = match family {
        AF_INET => text
            .parse::<std::net::Ipv4Addr>()
            .ok()
            .map(|a| a.octets().to_vec()),
        AF_INET6 => text
            .parse::<std::net::Ipv6Addr>()
            .ok()
            .map(|a| a.octets().to_vec()),
        _ => return env.fail(crate::errno::EAFNOSUPPORT),
    };
    match bytes {
        Some(b) => {
            env.mem.write(out, &b)?;
            Ok(Outcome::Return(1))
        }
        None => Ok(Outcome::Return(0)),
    }
}

/// `inet_ntop(family, addr, out, size)`: `out`, or null when it is too small.
fn inet_ntop(env: &mut Env) -> Result<Outcome, CallError> {
    let (family, addr, out, size) = (
        env.arg(0) as u32 as u64,
        env.arg(1),
        env.arg(2),
        env.arg(3) as u32 as usize,
    );
    let text = match family {
        AF_INET => {
            let mut b = [0u8; 4];
            env.mem.read(addr, &mut b)?;
            std::net::Ipv4Addr::from(b).to_string()
        }
        AF_INET6 => {
            let mut b = [0u8; 16];
            env.mem.read(addr, &mut b)?;
            std::net::Ipv6Addr::from(b).to_string()
        }
        _ => {
            env.set_errno(crate::errno::EAFNOSUPPORT)?;
            return Ok(Outcome::Return(0));
        }
    };
    if text.len() + 1 > size {
        env.set_errno(crate::errno::ENOSPC)?;
        return Ok(Outcome::Return(0));
    }
    let mut b = text.into_bytes();
    b.push(0);
    env.mem.write(out, &b)?;
    Ok(Outcome::Return(out))
}
