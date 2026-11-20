//! libSceAudioPropagation: acoustic propagation (rooms, portals, sources and
//! the paths sound takes between them). We model none of it: every call
//! succeeds and no source has an audio path, so a title plays its sounds
//! without propagation effects. Names hashed from SharpEmu-mac's
//! `AudioPropagationExports.cs` (GPL-2.0-or-later), which also gave the
//! layout of the memory request.

use crate::{CallError, Env, Function, Outcome};

/// Bytes of CPU memory we ask the title for in `SystemQueryMemory`.
const CPU_MEMORY: u64 = 1 << 20;
const ERROR_INVALID_POINTER: u64 = 0x8A70_0003;

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAudioPropagation",
        name,
        func,
    }
}

pub static STUBS: &[Function] = &[
    f("sceAudioPropagationSystemQueryMemory", query_memory),
    f("sceAudioPropagationSystemCreate", system_create),
    f("sceAudioPropagationSystemDestroy", ok),
    f("sceAudioPropagationSystemSetAttributes", ok),
    f("sceAudioPropagationSystemSetRays", ok),
    f("sceAudioPropagationSystemGetRays", ok),
    f("sceAudioPropagationSystemRegisterMaterial", ok),
    f("sceAudioPropagationSystemUnregisterMaterial", ok),
    f("sceAudioPropagationRoomCreate", ok),
    f("sceAudioPropagationRoomDestroy", ok),
    f("sceAudioPropagationPortalCreate", ok),
    f("sceAudioPropagationPortalDestroy", ok),
    f("sceAudioPropagationPortalSetAttributes", ok),
    f("sceAudioPropagationSourceCreate", ok),
    f("sceAudioPropagationSourceDestroy", ok),
    f("sceAudioPropagationSourceSetAttributes", ok),
    f("sceAudioPropagationSourceCalculateAudioPaths", ok),
    f("sceAudioPropagationSourceGetAudioPath", ok),
    f("sceAudioPropagationSourceGetAudioPathCount", ok),
    f("sceAudioPropagationSourceGetRays", ok),
    f("sceAudioPropagationSourceRender", ok),
    f("sceAudioPropagationSourceSetAudioPaths", ok),
];

/// `sceAudioPropagationSystemCreate(options, memory, *system)`: the system
/// handle is the CPU memory the title allocated for it (`memory` + 0x10),
/// so it is never 0.
fn system_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (memory, out) = (env.arg(1), env.arg(2));
    if memory == 0 || out == 0 {
        return Ok(Outcome::Return(ERROR_INVALID_POINTER));
    }
    let mut cpu = [0u8; 8];
    env.mem.read(memory + 0x10, &mut cpu)?;
    let handle = match u64::from_le_bytes(cpu) {
        0 => memory,
        p => p,
    };
    env.mem.write(out, &handle.to_le_bytes())?;
    Ok(Outcome::Return(0))
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

/// `sceAudioPropagationSystemQueryMemory(options, memory)`: `memory` holds a
/// struct descriptor (16 bytes), then CPU pointer and size, GPU pointer and
/// size; we fill in the two sizes and need no GPU memory.
fn query_memory(env: &mut Env) -> Result<Outcome, CallError> {
    let out = env.arg(1);
    if out == 0 {
        return Ok(Outcome::Return(ERROR_INVALID_POINTER));
    }
    env.mem.write(out + 0x18, &CPU_MEMORY.to_le_bytes())?;
    env.mem.write(out + 0x28, &0u64.to_le_bytes())?;
    Ok(Outcome::Return(0))
}
