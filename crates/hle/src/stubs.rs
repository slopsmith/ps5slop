//! Library set-up and tear-down calls that only need to succeed (or hand
//! back a fresh id) for a title to carry on: dialogs, save data, network
//! contexts. Each is a placeholder until a title needs the real behaviour.
//!
//! `#qBS714-Jr3g` (libSceLibcInternalExt, name unknown) is what Minecraft's
//! `thread_local` set-up calls with a destructor, an object and a DSO
//! handle, as `__cxa_thread_atexit` takes; thread-exit destructors are not
//! run yet, so it only succeeds. `#YFC3dBBipj8` (libkernel_write_throttling)
//! fills a write budget in 64 KiB units, judging by Minecraft's use; failing
//! makes Minecraft use its own default.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{CallError, Env, Function, Outcome};

const fn zero(library: &'static str, name: &'static str) -> Function {
    Function {
        library,
        name,
        func: ok,
    }
}

const fn id(library: &'static str, name: &'static str) -> Function {
    Function {
        library,
        name,
        func: new_id,
    }
}

/// Writes a fresh id (32 bits) to the pointer in argument `N`.
const fn out_id<const N: usize>(library: &'static str, name: &'static str) -> Function {
    Function {
        library,
        name,
        func: write_id::<N>,
    }
}

fn write_id<const N: usize>(env: &mut Env) -> Result<Outcome, CallError> {
    let id = NEXT.fetch_add(1, Ordering::Relaxed) as u32;
    if env.arg(N) != 0 {
        env.mem.write(env.arg(N), &id.to_le_bytes())?;
    }
    Ok(Outcome::Return(0))
}

fn unsupported(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(crate::dmem::sce_error(
        crate::errno::ENOSYS,
    )))
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

static NEXT: AtomicU64 = AtomicU64::new(1);

/// A positive id, unique across these calls.
fn new_id(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(NEXT.fetch_add(1, Ordering::Relaxed)))
}

pub static FUNCTIONS: &[Function] = &[
    Function {
        library: "libSceAppContent",
        name: "sceAppContentTemporaryDataMount2",
        func: temporary_data_mount,
    },
    Function {
        library: "libSceAppContent",
        name: "sceAppContentTemporaryDataGetAvailableSpaceKb",
        func: temporary_data_space,
    },
    Function {
        library: "libSceAppContent",
        name: "sceAppContentAppParamGetInt",
        func: app_param_int,
    },
    Function {
        library: "libSceRandom",
        name: "sceRandomGetRandomNumber",
        func: random_number,
    },
    Function {
        library: "libkernel",
        name: "sceKernelUuidCreate",
        func: uuid_create,
    },
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    zero("libSceCommonDialog", "sceCommonDialogInitialize"),
    zero("libSceMsgDialog.native", "sceMsgDialogInitialize"),
    zero("libSceMsgDialog.native", "sceMsgDialogTerminate"),
    zero("libSceSaveDataDialog.native", "sceSaveDataDialogInitialize"),
    zero("libSceSaveDataDialog.native", "sceSaveDataDialogTerminate"),
    zero("libSceSigninDialog", "sceSigninDialogInitialize"),
    zero("libSceSigninDialog", "sceSigninDialogTerminate"),
    zero("libSceWebBrowserDialog", "sceWebBrowserDialogInitialize"),
    zero("libSceWebBrowserDialog", "sceWebBrowserDialogTerminate"),
    zero("libSceNpCommerce", "sceNpCommerceDialogInitialize"),
    zero("libSceNpCommerce", "sceNpCommerceDialogTerminate"),
    zero("libSceImeDialog", "sceImeDialogInit"),
    zero("libSceImeDialog", "sceImeDialogTerm"),
    zero("libSceMouse", "sceMouseInit"),
    id("libSceMouse", "sceMouseOpen"),
    zero("libSceMouse", "sceMouseClose"),
    zero("libSceMouse", "sceMouseRead"),
    zero("libSceIme", "sceImeKeyboardOpen"),
    zero("libSceIme", "sceImeKeyboardClose"),
    zero("libSceIme", "sceImeUpdate"),
    zero("libSceAppContent", "sceAppContentInitialize"),
    zero("libSceSaveData_native", "sceSaveDataInitialize3"),
    zero("libSceSaveData_native", "sceSaveDataTerminate"),
    id("libSceSsl", "sceSslInit"),
    zero("libSceSsl", "sceSslTerm"),
    id("libSceHttp", "sceHttpInit"),
    zero("libSceHttp", "sceHttpTerm"),
    id("libSceHttp", "sceHttpCreateTemplate"),
    id("libSceHttp2", "sceHttp2Init"),
    zero("libSceHttp2", "sceHttp2Term"),
    id("libSceHttp2", "sceHttp2CreateTemplate"),
    zero("libSceHttp2", "sceHttp2DeleteTemplate"),
    id("libSceNet", "sceNetResolverCreate"),
    id("libSceNpManager", "sceNpCreateRequest"),
    id("libSceNpAuth", "sceNpAuthCreateRequest"),
    zero(
        "libSceSystemService",
        "sceSystemServiceReportAbnormalTermination",
    ),
    id("libSceNpWebApi2", "sceNpWebApi2Initialize"),
    zero("libSceNpWebApi2", "sceNpWebApi2Terminate"),
    id("libSceNpWebApi2", "sceNpWebApi2CreateUserContext"),
    zero("libSceNpWebApi2", "sceNpWebApi2DeleteUserContext"),
    id("libSceNpWebApi2", "sceNpWebApi2PushEventCreateHandle"),
    zero("libSceNpWebApi2", "sceNpWebApi2PushEventDeleteHandle"),
    id("libSceNpWebApi2", "sceNpWebApi2PushEventCreateFilter"),
    zero("libSceNpWebApi2", "sceNpWebApi2PushEventDeleteFilter"),
    id("libSceNpWebApi2", "sceNpWebApi2PushEventRegisterCallback"),
    zero("libSceNpWebApi2", "sceNpWebApi2PushEventUnregisterCallback"),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemInitialize",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemTerminate",
    ),
    out_id::<0>(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateContext",
    ),
    out_id::<0>(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateHandle",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemRegisterContext",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemDestroyHandle",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemAbortHandle",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemPostEvent",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemDestroyContext",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemDestroyEvent",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyObjectSetString",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyObjectSetInt32",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyObjectSetArray",
    ),
    zero(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyArraySetString",
    ),
    zero("libSceNpGameIntent", "sceNpGameIntentInitialize"),
    zero("libSceNpGameIntent", "sceNpGameIntentTerminate"),
    out_id::<0>("libSceNpTrophy2", "sceNpTrophy2CreateContext"),
    out_id::<0>("libSceNpTrophy2", "sceNpTrophy2CreateHandle"),
    zero(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessInitialize",
    ),
    id("libSceNpManager", "sceNpRegisterStateCallbackA"),
    zero("libSceNpManager", "sceNpCheckCallback"),
    out_id::<2>("libSceNetCtl", "sceNetCtlRegisterCallback"),
    zero("libSceNetCtl", "sceNetCtlCheckCallback"),
    zero("libSceAudioOut2", "sceAudioOut2Initialize"),
    zero("libSceCoredump", "sceCoredumpRegisterCoredumpHandler"),
    zero("libSceCoredump", "sceCoredumpUnregisterCoredumpHandler"),
    zero("libSceTextToSpeech2", "sceTextToSpeech2Initialize"),
    zero("libSceTextToSpeech2", "sceTextToSpeech2Terminate"),
    zero(
        "libSceNpAuthAuthorizedAppDialog",
        "sceNpAuthAuthorizedAppDialogInitialize",
    ),
    zero(
        "libSceNpAuthAuthorizedAppDialog",
        "sceNpAuthAuthorizedAppDialogTerminate",
    ),
    out_id::<1>("libSceAjm", "sceAjmInitialize"),
    zero("libSceAjm", "sceAjmModuleRegister"),
    out_id::<3>("libSceAjm", "sceAjmInstanceCreate"),
    Function {
        library: "libSceNpManager",
        name: "sceNpGetState",
        func: np_state,
    },
    Function {
        library: "libSceNetCtl",
        name: "sceNetCtlGetState",
        func: netctl_state,
    },
    Function {
        library: "libSceLibcInternalExt",
        name: "#qBS714-Jr3g",
        func: ok,
    },
    Function {
        library: "libkernel_write_throttling",
        name: "#YFC3dBBipj8",
        func: unsupported,
    },
];

/// `sceKernelUuidCreate(uuid)`: a random (version 4) UUID.
fn uuid_create(env: &mut Env) -> Result<Outcome, CallError> {
    let mut b = [0u8; 16];
    let mut x = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
        ^ NEXT_UUID.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
    for chunk in b.chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        chunk.copy_from_slice(&x.to_le_bytes());
    }
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    env.mem.write(env.arg(0), &b)?;
    Ok(Outcome::Return(0))
}

static NEXT_UUID: AtomicU64 = AtomicU64::new(1);

/// `sceNpGetState(user, &state)`: signed out (state 1, unverified).
fn np_state(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(1), &1u32.to_le_bytes())?;
    Ok(Outcome::Return(0))
}

/// `sceNetCtlGetState(&state)`: disconnected (state 0, unverified).
fn netctl_state(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(0), &0u32.to_le_bytes())?;
    Ok(Outcome::Return(0))
}

/// `sceRandomGetRandomNumber(buf, size)`: host random bytes, at most 64
/// (the limit shadPS4 uses; unverified).
fn random_number(env: &mut Env) -> Result<Outcome, CallError> {
    use std::io::Read;
    let (buf, size) = (env.arg(0), env.arg(1) as usize);
    if size > 64 {
        return Ok(Outcome::Return(0x817C_0016));
    }
    let mut b = vec![0u8; size];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|_| CallError::Abort("cannot read /dev/urandom"))?;
    env.mem.write(buf, &b)?;
    Ok(Outcome::Return(0))
}

/// `sceAppContentTemporaryDataMount2(option, mount_point)`: the temporary
/// data folder is always mounted at `/temp0` (see `main.rs`).
fn temporary_data_mount(env: &mut Env) -> Result<Outcome, CallError> {
    let mut point = [0u8; 16];
    point[..6].copy_from_slice(b"/temp0");
    env.mem.write(env.arg(1), &point)?;
    Ok(Outcome::Return(0))
}

/// `sceAppContentTemporaryDataGetAvailableSpaceKb(mount_point, &kb)`:
/// 1 GiB (our choice).
fn temporary_data_space(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write_u64(env.arg(1), 1 << 20)?;
    Ok(Outcome::Return(0))
}

/// `sceAppContentAppParamGetInt(id, &value)`: the full game (SKU flag 3),
/// and 0 for the four user-defined parameters (encodings unverified).
fn app_param_int(env: &mut Env) -> Result<Outcome, CallError> {
    let value: i32 = match env.arg(0) as u32 {
        0 => 3,
        1..=4 => 0,
        _ => {
            return Ok(Outcome::Return(crate::dmem::sce_error(
                crate::errno::EINVAL,
            )));
        }
    };
    env.mem.write(env.arg(1), &value.to_le_bytes())?;
    Ok(Outcome::Return(0))
}
