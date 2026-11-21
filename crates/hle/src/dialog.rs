//! System dialogs (libSceMsgDialog, libSceSigninDialog, libSceSaveDataDialog). We show nothing: an
//! opened dialog finishes at once, a message dialog as if the user pressed
//! its first button, a sign-in dialog as if the user cancelled. Message texts
//! go to stderr. Status values, results and the PS4 parameter layouts come
//! from the OpenOrbis headers (`docs/hle-kernel.md`); the PS5 is unverified.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::{CallError, Env, Function, Outcome};

const NONE: u32 = 0;
const INITIALIZED: u32 = 1;
const FINISHED: u32 = 3;
const RESULT_OK: u32 = 0;
const RESULT_USER_CANCELED: u32 = 1;
const BUTTON_OK: u32 = 1;

/// `OrbisMsgDialogParam`: the mode, and the user message parameters.
const PARAM_MODE: u64 = 56;
const PARAM_USER_MSG: u64 = 64;
/// `OrbisMsgDialogUserMessageParam`: the message text.
const USER_MSG_TEXT: u64 = 8;
const MODE_USER_MSG: u32 = 1;

static MSG: AtomicU32 = AtomicU32::new(NONE);
static MSG_MODE: AtomicU32 = AtomicU32::new(0);
static SIGNIN: AtomicU32 = AtomicU32::new(NONE);
static SAVE: AtomicU32 = AtomicU32::new(NONE);

const fn f(library: &'static str, name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library,
        name,
        func,
    }
}

pub static FUNCTIONS: &[Function] = &[
    f("libSceMsgDialog.native", "sceMsgDialogInitialize", |_| {
        set(&MSG, INITIALIZED)
    }),
    f("libSceMsgDialog.native", "sceMsgDialogTerminate", |_| {
        set(&MSG, NONE)
    }),
    f("libSceMsgDialog.native", "sceMsgDialogOpen", msg_open),
    f("libSceMsgDialog.native", "sceMsgDialogUpdateStatus", |_| {
        status(&MSG)
    }),
    f("libSceMsgDialog.native", "sceMsgDialogGetStatus", |_| {
        status(&MSG)
    }),
    f(
        "libSceMsgDialog.native",
        "sceMsgDialogGetResult",
        msg_result,
    ),
    f("libSceMsgDialog.native", "sceMsgDialogClose", |_| {
        set(&MSG, FINISHED)
    }),
    f("libSceSigninDialog", "sceSigninDialogInitialize", |_| {
        set(&SIGNIN, INITIALIZED)
    }),
    f("libSceSigninDialog", "sceSigninDialogTerminate", |_| {
        set(&SIGNIN, NONE)
    }),
    f("libSceSigninDialog", "sceSigninDialogOpen", signin_open),
    f("libSceSigninDialog", "sceSigninDialogUpdateStatus", |_| {
        status(&SIGNIN)
    }),
    f("libSceSigninDialog", "sceSigninDialogGetStatus", |_| {
        status(&SIGNIN)
    }),
    f(
        "libSceSigninDialog",
        "sceSigninDialogGetResult",
        signin_result,
    ),
    f("libSceSigninDialog", "sceSigninDialogClose", |_| {
        set(&SIGNIN, FINISHED)
    }),
    f(
        "libSceSaveDataDialog.native",
        "sceSaveDataDialogInitialize",
        |_| set(&SAVE, INITIALIZED),
    ),
    f(
        "libSceSaveDataDialog.native",
        "sceSaveDataDialogTerminate",
        |_| set(&SAVE, NONE),
    ),
    f(
        "libSceSaveDataDialog.native",
        "sceSaveDataDialogOpen",
        save_open,
    ),
    f(
        "libSceSaveDataDialog.native",
        "sceSaveDataDialogUpdateStatus",
        |_| status(&SAVE),
    ),
    f(
        "libSceSaveDataDialog.native",
        "sceSaveDataDialogGetStatus",
        |_| status(&SAVE),
    ),
    f(
        "libSceSaveDataDialog.native",
        "sceSaveDataDialogGetResult",
        |_| Ok(Outcome::Return(0)),
    ),
    f(
        "libSceSaveDataDialog.native",
        "sceSaveDataDialogClose",
        |_| set(&SAVE, FINISHED),
    ),
    f(
        "libSceSaveDataDialog.native",
        "sceSaveDataDialogProgressBarSetValue",
        |_| Ok(Outcome::Return(0)),
    ),
];

fn set(state: &AtomicU32, to: u32) -> Result<Outcome, CallError> {
    state.store(to, Ordering::Relaxed);
    Ok(Outcome::Return(0))
}

fn status(state: &AtomicU32) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(state.load(Ordering::Relaxed) as u64))
}

/// `sceMsgDialogOpen(param)`.
fn msg_open(env: &mut Env) -> Result<Outcome, CallError> {
    let param = env.arg(0);
    let mut mode = [0u8; 4];
    env.mem.read(param + PARAM_MODE, &mut mode)?;
    let mode = u32::from_le_bytes(mode);
    MSG_MODE.store(mode, Ordering::Relaxed);
    let text = match mode {
        MODE_USER_MSG => env
            .mem
            .read_u64(param + PARAM_USER_MSG)
            .and_then(|p| env.mem.read_u64(p + USER_MSG_TEXT))
            .and_then(|t| env.mem.read_cstr(t, 1024))
            .map(|t| String::from_utf8_lossy(&t).into_owned())
            .unwrap_or_default(),
        _ => String::new(),
    };
    eprintln!("ps5slop: message dialog (mode {mode}): {text:?}");
    set(&MSG, FINISHED)
}

/// `sceMsgDialogGetResult(result)`: `{mode, result, buttonId}`.
fn msg_result(env: &mut Env) -> Result<Outcome, CallError> {
    let mut r = [0u8; 12];
    r[0..4].copy_from_slice(&MSG_MODE.load(Ordering::Relaxed).to_le_bytes());
    r[4..8].copy_from_slice(&RESULT_OK.to_le_bytes());
    r[8..12].copy_from_slice(&BUTTON_OK.to_le_bytes());
    env.mem.write(env.arg(0), &r)?;
    Ok(Outcome::Return(0))
}

/// `sceSaveDataDialogOpen(param)`: its result is left as the caller
/// initialised it, since we know no layout for it.
fn save_open(_: &mut Env) -> Result<Outcome, CallError> {
    eprintln!("ps5slop: save data dialog opened; finished at once");
    set(&SAVE, FINISHED)
}

/// `sceSigninDialogOpen(param)`.
fn signin_open(_: &mut Env) -> Result<Outcome, CallError> {
    eprintln!("ps5slop: sign-in dialog opened; answering \"cancelled\"");
    set(&SIGNIN, FINISHED)
}

/// `sceSigninDialogGetResult(result)`: the result comes first (layout
/// unverified).
fn signin_result(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem
        .write(env.arg(0), &RESULT_USER_CANCELED.to_le_bytes())?;
    Ok(Outcome::Return(0))
}

pub(crate) fn save(e: &mut savestate::Encoder) {
    e.put(&[&MSG, &MSG_MODE, &SIGNIN, &SAVE].map(|a| a.load(Ordering::Relaxed)));
}

pub(crate) fn load(d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
    let v: [u32; 4] = d.get()?;
    for (a, v) in [&MSG, &MSG_MODE, &SIGNIN, &SAVE].into_iter().zip(v) {
        a.store(v, Ordering::Relaxed);
    }
    Ok(())
}
