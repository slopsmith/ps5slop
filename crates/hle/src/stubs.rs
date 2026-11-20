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

/// Writes a 32-bit zero (an empty count) to the pointer in argument `N`.
fn write_zero_u32<const N: usize>(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(N) != 0 {
        env.mem.write(env.arg(N), &0u32.to_le_bytes())?;
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
    // We are a base PS4, not a PS4 Pro, to PS4 titles.
    zero("libkernel", "sceKernelIsNeoMode"),
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
        name: "sceAppContentDownloadDataGetAvailableSpaceKb",
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
    Function {
        library: "libkernel",
        name: "#crb5j7mkk1c",
        func: unsupported,
    },
    Function {
        library: "libSceIme",
        name: "sceImeKeyboardGetResourceId",
        func: unsupported,
    },
    Function {
        library: "libSceNpCppWebApi",
        name: "#Y295ygEccqk",
        func: unsupported,
    },
    Function {
        library: "libSceNpCppWebApi",
        name: "#UYPxv8MIzGo",
        func: unsupported,
    },
    Function {
        library: "libSceTextToSpeech2",
        name: "#08JSg9p6bgQ",
        func: unsupported,
    },
    Function {
        library: "libSceNpAuth",
        name: "sceNpAuthGetAuthorizationCodeV3",
        func: unsupported,
    },
    Function {
        library: "libSceNpAuth",
        name: "#IDX0S5EsEh4",
        func: unsupported,
    },
    Function {
        library: "libSceNpAuth",
        name: "sceNpAuthDeleteRequest",
        func: ok,
    },
    Function {
        library: "libSceNpManager",
        name: "sceNpDeleteRequest",
        func: ok,
    },
    Function {
        library: "libSceNpManager",
        name: "sceNpUnregisterStateCallbackA",
        func: ok,
    },
    Function {
        library: "libSceNpManager",
        name: "sceNpGetAccountIdA",
        func: unsupported,
    },
    Function {
        library: "libSceNpManager",
        name: "sceNpCheckPremium",
        func: unsupported,
    },
    Function {
        library: "libSceNpManager",
        name: "sceNpNotifyPremiumFeature",
        func: ok,
    },
    Function {
        library: "libSceNpManager",
        name: "sceNpCheckNpReachability",
        func: unsupported,
    },
    Function {
        library: "libSceNpManager",
        name: "sceNpGetNpReachabilityState",
        func: unsupported,
    },
    Function {
        library: "libSceNpEntitlementAccess",
        name: "sceNpEntitlementAccessGetSkuFlag",
        func: sku_flag,
    },
    Function {
        library: "libSceAcm",
        name: "sceAcmContextCreate",
        func: unsupported,
    },
    Function {
        library: "libSceTextToSpeech2",
        name: "sceTextToSpeech2Open",
        func: unsupported,
    },
    Function {
        library: "libSceTextToSpeech2",
        name: "sceTextToSpeech2Speak",
        func: unsupported,
    },
    Function {
        library: "libSceTextToSpeech2",
        name: "sceTextToSpeech2Cancel",
        func: ok,
    },
    Function {
        library: "libSceTextToSpeech2",
        name: "sceTextToSpeech2Close",
        func: ok,
    },
    Function {
        library: "libSceHttp2",
        name: "sceHttp2CreateRequestWithURL",
        func: unsupported,
    },
    Function {
        library: "libSceHttp",
        name: "sceHttpCreateConnectionWithURL",
        func: unsupported,
    },
    Function {
        library: "libSceHttp",
        name: "sceHttpCreateRequestWithURL",
        func: unsupported,
    },
    Function {
        library: "libSceNpWebApi2",
        name: "sceNpWebApi2CreateRequest",
        func: unsupported,
    },
    zero("libkernel", "sceKernelGetGPI"),
    zero("libkernel", "#n1-v6FgU7MQ"),
    zero(
        "libSceSystemService",
        "sceSystemServiceGetHdrToneMapLuminance",
    ),
    zero(
        "libSceSystemService",
        "sceSystemServiceGetNoticeScreenSkipFlag",
    ),
    zero("libScePad", "scePadGetTriggerEffectState"),
    Function {
        library: "libSceNpEntitlementAccess",
        name: "sceNpEntitlementAccessGetAddcontEntitlementInfoList",
        func: write_zero_u32::<3>,
    },
    zero(
        "libSceUserServicePlatformPrivacyWs1",
        "sceUserServiceGetPlatformPrivacyWs1",
    ),
    zero("libSceShare", "sceShareInitialize"),
    zero("libSceShare", "sceShareRegisterContentEventCallback"),
    zero("libSceContentExport", "sceContentExportInit2"),
    zero("libSceNetCtl", "sceNetCtlInit"),
    zero("libSceNetCtl", "sceNetCtlTerm"),
    zero("libSceRudp", "sceRudpInit"),
    zero("libSceRudp", "sceRudpSetEventHandler"),
    zero("libSceRudp", "sceRudpEnableInternalIOThread"),
    zero("libScePlayGo", "scePlayGoInitialize"),
    out_id::<0>("libScePlayGo", "scePlayGoOpen"),
    zero("libScePlayGo", "scePlayGoSetInstallSpeed"),
    Function {
        library: "libScePlayGo",
        name: "scePlayGoGetChunkId",
        func: write_zero_u32::<3>,
    },
    Function {
        library: "libScePlayGo",
        name: "scePlayGoGetToDoList",
        func: write_zero_u32::<3>,
    },
    zero("libScePlayGo", "scePlayGoClose"),
    zero("libScePlayGo", "scePlayGoTerminate"),
    zero("libSceSaveData_native", "sceSaveDataSetupSaveDataMemory2"),
    zero("libSceSaveData_native", "sceSaveDataGetSaveDataMemory2"),
    zero("libSceSaveData_native", "sceSaveDataSetSaveDataMemory2"),
    zero("libSceSaveData_native", "sceSaveDataSyncSaveDataMemory"),
    zero(
        "libSceJson2",
        "_ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE",
    ),
    zero("libSceJson2", "_ZN3sce4Json5ValueC1Ev"),
    zero("libSceJson2", "_ZN3sce4Json5ValueD1Ev"),
    zero("libSceJson2", "_ZN3sce4Json5Value3setEb"),
    zero("libSceJson2", "_ZN3sce4Json5Value3setEl"),
    zero("libSceJson2", "_ZN3sce4Json5Value3setEm"),
    zero("libSceJson2", "_ZN3sce4Json5Value3setEd"),
    zero("libSceJson2", "_ZN3sce4Json5Value3setENS0_9ValueTypeE"),
    zero("libSceJson2", "_ZN3sce4Json6StringC1EPKc"),
    zero("libSceJson2", "_ZN3sce4Json6StringD1Ev"),
    zero("libSceJson2", "#6l3Bv2gysNc"),
    zero("libSceJson2", "#00oCq0RwSAY"),
    zero("libSceNpCppWebApi", "#oTirsxQpqj0"),
    zero("libSceAgc", "#dolOmWH+huQ"),
    zero("libSceAgc", "#fd5Bp5tGTgo"),
    zero("libSceCommonDialog", "sceCommonDialogInitialize"),
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
    zero("libSceIme", "sceImeSetText"),
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
    zero("libSceNpTrophy2", "sceNpTrophy2RegisterContext"),
    out_id::<2>(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateEvent",
    ),
    zero("libSceNpTrophy2", "sceNpTrophy2DestroyHandle"),
    zero("libSceNpTrophy2", "sceNpTrophy2DestroyContext"),
    zero(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessInitialize",
    ),
    id("libSceNpManager", "sceNpRegisterStateCallbackA"),
    zero("libSceNpManager", "sceNpCheckCallback"),
    zero("libSceNpManager", "sceNpSetNpTitleId"),
    out_id::<2>("libSceNetCtl", "sceNetCtlRegisterCallback"),
    zero("libSceNetCtl", "sceNetCtlCheckCallback"),
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
    zero("libSceNpCppWebApi", "#8x++mBOUeso"),
    zero("libSceNpCppWebApi", "#52AlYvq+dmk"),
    zero("libSceJson2", "_ZN3sce4Json11InitializerC1Ev"),
    zero("libSceJson2", "_ZN3sce4Json14InitParameter2C1Ev"),
    zero(
        "libSceJson2",
        "_ZN3sce4Json14InitParameter212setAllocatorEPNS0_12MemAllocatorEPv",
    ),
    zero(
        "libSceJson2",
        "_ZN3sce4Json14InitParameter217setFileBufferSizeEm",
    ),
    zero("libSceJson2", "_ZN3sce4Json11InitializerD1Ev"),
    zero("libSceJson2", "_ZN3sce4Json12MemAllocatorC2Ev"),
    zero("libSceJson2", "_ZN3sce4Json12MemAllocatorD2Ev"),
    Function {
        library: "libSceJson2",
        name: "_ZN3sce4Json11Initializer10initializeEPKNS0_14InitParameter2E",
        func: unsupported,
    },
    out_id::<3>("libSceAjm", "sceAjmInstanceCreate"),
    Function {
        library: "libSceNpManager",
        name: "sceNpGetState",
        func: np_state,
    },
    Function {
        library: "libSceNpManager",
        name: "sceNpGetAccountCountryA",
        func: np_account_country,
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
    // Name unknown; Deltarune calls it once at start-up and carries on when
    // it succeeds.
    Function {
        library: "libkernel_cpumode_platform",
        name: "#mpxAdqW7dKY",
        func: ok,
    },
    Function {
        library: "libkernel_write_throttling",
        name: "#YFC3dBBipj8",
        func: unsupported,
    },
    Function {
        library: "libSceNpTrophy2",
        name: "sceNpTrophy2GetGameInfo",
        func: unsupported,
    },
    Function {
        library: "libSceNpTrophy2",
        name: "sceNpTrophy2GetTrophyInfo",
        func: unsupported,
    },
    Function {
        library: "libSceNpCppWebApi",
        name: "#Y295ygEccqk",
        func: unsupported,
    },
    Function {
        library: "libSceNpCppWebApi",
        name: "#52AlYvq+dmk",
        func: unsupported,
    },
    Function {
        library: "libSceNpCppWebApi",
        name: "#UYPxv8MIzGo",
        func: unsupported,
    },
    Function {
        library: "libSceNpCppWebApi",
        name: "#8x++mBOUeso",
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

/// `sceNpGetAccountCountryA(user, &country)`: "us", as an ISO 3166-1 code
/// with a terminating NUL (layout and signed-out behaviour unverified).
fn np_account_country(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(1), b"us\0\0")?;
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

/// `sceNpEntitlementAccessGetSkuFlag(&flag)`: the full game (3, unverified).
fn sku_flag(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(0), &3u32.to_le_bytes())?;
    Ok(Outcome::Return(0))
}
