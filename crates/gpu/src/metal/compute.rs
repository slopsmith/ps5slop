//! Dispatches: the compute path, indirect sizes and the loop guard.

use super::*;

impl Metal {
    /// Encodes a one-thread program that reads the dimensions at `at` once
    /// earlier work is done and writes threadgroup counts and, 16 bytes on,
    /// the grid size in threads, into the returned buffer.
    fn indirect_sizes(
        &mut self,
        host: &dyn Host,
        at: u64,
        size: [usize; 3],
        threads: bool,
    ) -> Option<(Retained<Buffer>, usize)> {
        let (start, end) = host.gpu_area()?;
        if !at.is_multiple_of(4) || at < start || at + 12 > end {
            return None;
        }
        let (dims, dims_at) = self.memory_buffer(host, at, 12)?;
        let (out, out_at) = self.scratch(32)?;
        if self.indirect_sizes.is_none() {
            let msl = INDIRECT_MSL.replace("LIMIT", &crate::pm4::MAX_GROUPS.to_string());
            let lib = self
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(&msl), None)
                .ok()?;
            let f = lib.newFunctionWithName(&NSString::from_str("indirect_sizes"))?;
            self.indirect_sizes = Some(
                self.device
                    .newComputePipelineStateWithFunction_error(&f)
                    .ok()?,
            );
        }
        let pipeline = self.indirect_sizes.clone()?;
        self.end_pass();
        let command = self.command();
        let enc = command.computeCommandEncoder()?;
        if let Some(f) = self.wait_memory() {
            enc.waitForFence(f);
        }
        enc.setComputePipelineState(&pipeline);
        let p = [
            size[0] as u32,
            size[1] as u32,
            size[2] as u32,
            threads as u32,
        ];
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&dims), dims_at, 0);
            enc.setBuffer_offset_atIndex(Some(&out), out_at, 1);
            enc.setBytes_length_atIndex(NonNull::new(p.as_ptr() as *mut _)?, 16, 2);
        }
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        enc.endEncoding();
        Some((out, out_at))
    }

    /// A zeroed slot of loop guard words for the next dispatch, as a byte offset.
    fn guard_slot(&mut self, program: u64, user: &[u32]) -> Option<(Retained<Buffer>, usize)> {
        if self.guard.is_none() {
            self.guard = Some(self.device.newBufferWithLength_options(
                GUARD_SLOTS * 8,
                MTLResourceOptions::StorageModeShared,
            )?);
        }
        let buffer = self.guard.clone()?;
        if self.guarded.len() >= GUARD_SLOTS / 2 {
            self.check_guards();
        }
        let slot = self.guard_next;
        self.guard_next = (slot + 1) % GUARD_SLOTS;
        // SAFETY: a shared buffer stays mapped; the dispatch that used this
        // slot before is long done, and a late one could only gain budget.
        unsafe {
            buffer
                .contents()
                .as_ptr()
                .cast::<u64>()
                .add(slot)
                .write_volatile(0)
        };
        let mut words = [0; 16];
        words.copy_from_slice(&user[..16]);
        self.guarded.push((slot, program, words));
        Some((buffer, slot * 8))
    }

    /// Reports dispatches whose loops gave up since the last check.
    pub(super) fn check_guards(&mut self) {
        let Some(buffer) = &self.guard else { return };
        let words = buffer.contents().as_ptr().cast::<u32>();
        for (slot, program, user) in std::mem::take(&mut self.guarded) {
            // SAFETY: the slot is inside the shared buffer, which stays mapped.
            let (units, gave_up) = unsafe {
                (
                    words.add(2 * slot).read_volatile(),
                    words.add(2 * slot + 1).read_volatile(),
                )
            };
            if gave_up != 0 {
                self.guard_trips += 1;
                diag::problems::record_with_detail(
                    diag::problems::Kind::LoopGuard,
                    "compute loop guard",
                    format!("cs {program:#x}"),
                    format!(
                        "{gave_up} threads, {units}/{} branch units",
                        gfx10::translate::GUARD_BUDGET
                    ),
                );
                if self.guard_trips <= 16 {
                    tracing::warn!(
                        "compute program {program:#x}: {gave_up} threads gave up their loops ({units} of {} branch units), user data {user:x?}",
                        gfx10::translate::GUARD_BUDGET
                    );
                }
            }
        }
    }

    pub(super) fn try_dispatch(
        &mut self,
        host: &dyn Host,
        state: &State,
        grid: Grid,
    ) -> Result<(), DispatchError> {
        let (groups, grid_bounds) = match grid {
            Grid::Direct(d) => (d.groups, false),
            Grid::Indirect { threads, .. } => ([0; 3], threads),
        };
        let s = state.sh;
        let address =
            ((s[sh::COMPUTE_PGM_LO] as u64) | (s[sh::COMPUTE_PGM_LO + 1] as u64) << 32) << 8;
        if address == 0 {
            return Err(DispatchError::Skipped("no compute program"));
        }
        if skipped(address) {
            return Err(DispatchError::Skipped("skipped by gpuskip"));
        }
        let users = (s[sh::COMPUTE_PGM_RSRC2] >> 1) & 0x1F;
        let user =
            &s[sh::COMPUTE_USER_DATA_0..sh::COMPUTE_USER_DATA_0 + sh::COMPUTE_USER_DATA_COUNT];
        self.note(|| {
            format!(
                "compute {address:#x} groups {groups:?} user {:x?}",
                &user[..users as usize]
            )
        });
        let mut fetch = |grid_bounds| {
            self.typed_program(
                host,
                state,
                Stage::Compute,
                address,
                users,
                false,
                user,
                Variant {
                    grid_bounds,
                    ..Variant::default()
                },
            )
            .ok_or(if grid_bounds {
                DispatchError::FallbackCpu
            } else {
                DispatchError::Skipped("compute program")
            })
        };
        let mut program = fetch(grid_bounds)?;
        let partial = match grid {
            Grid::Direct(d) => d.threads,
            Grid::Indirect { .. } => None,
        };
        let mut whole = partial.filter(|_| program.translated.lockstep);
        if whole.is_some() {
            match fetch(true) {
                Ok(p) => program = p,
                Err(_) => whole = None,
            }
        }
        self.trace_user_memory(host, &user[..users as usize], false);
        let task = program
            .compute
            .as_ref()
            .ok_or(DispatchError::Skipped("compute pipeline"))?;
        if let Grid::Direct(d) = grid
            && (groups.contains(&0) || d.threads.unwrap_or(groups).contains(&0))
        {
            return Ok(());
        }
        let pipeline = match task.poll() {
            Some(p) => p,
            None => {
                let t = std::time::Instant::now();
                let p = task.wait_blocking();
                self.note_wait(1, t);
                p
            }
        }
        .ok_or(DispatchError::Skipped("compute pipeline"))?;
        let threads = [0, 1, 2].map(|i| (s[sh::COMPUTE_NUM_THREAD_X + i] & 0xFFFF).max(1) as usize);
        if threads.iter().product::<usize>() > pipeline.maxTotalThreadsPerThreadgroup() {
            return Err(DispatchError::Skipped("threadgroup too large"));
        }
        let bound = self.resolve(host, &program, user);
        let sizes = match grid {
            Grid::Indirect { at, threads: t } if bound.is_some() => Some(
                self.indirect_sizes(host, at, threads, t)
                    .ok_or(DispatchError::FallbackCpu)?,
            ),
            _ => None,
        };
        self.end_pass();
        let command = self.command();
        let enc = command
            .computeCommandEncoder()
            .ok_or(DispatchError::Skipped("compute encoder"))?;
        if crate::timing::labels() {
            enc.setLabel(Some(&NSString::from_str(&format!(
                "dispatch {:#x} {groups:?}",
                program.address
            ))));
        }
        let memory = program.translated.guest_memory && bound.is_some();
        if memory && let Some(f) = self.wait_memory() {
            enc.waitForFence(f);
        }
        enc.setComputePipelineState(&pipeline);
        let guard = if program.translated.loop_guard {
            self.guard_slot(program.address, user)
        } else {
            None
        };
        if program.translated.loop_guard && guard.is_none() {
            enc.endEncoding();
            return Err(DispatchError::Skipped("loop guard buffer"));
        }
        if let Some(b) = &bound {
            Self::apply(Slot::Compute(&enc), b, user);
            if let Some((g, offset)) = &guard {
                Slot::Compute(&enc).buffer(g, *offset, binding::GUARD as usize);
            }
            let per_group = MTLSize {
                width: threads[0],
                height: threads[1],
                depth: threads[2],
            };
            match grid {
                Grid::Indirect { threads, .. } => {
                    if let Some((args, at)) = &sizes {
                        if threads {
                            Slot::Compute(&enc).buffer(args, at + 16, binding::BOUNDS as usize);
                        }
                        unsafe {
                            enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                                args, *at, per_group,
                            )
                        }
                    }
                }
                Grid::Direct(_) if let Some(t) = whole => {
                    Slot::Compute(&enc).bytes(
                        t.as_ptr().cast(),
                        size_of_val(&t),
                        binding::BOUNDS as usize,
                    );
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups[0] as usize,
                            height: groups[1] as usize,
                            depth: groups[2] as usize,
                        },
                        per_group,
                    )
                }
                Grid::Direct(crate::Dispatch {
                    threads: Some(t), ..
                }) => enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: t[0] as usize,
                        height: t[1] as usize,
                        depth: t[2] as usize,
                    },
                    per_group,
                ),
                Grid::Direct(_) => enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups[0] as usize,
                        height: groups[1] as usize,
                        depth: groups[2] as usize,
                    },
                    per_group,
                ),
            }
            self.note_writes(host, &program.translated, user);
            self.pending_writes |= program.translated.writes_memory;
        }
        if memory
            && program.translated.writes_memory
            && let Some(f) = self.fence.as_deref()
        {
            enc.updateFence(f);
            self.fence_updated = true;
        }
        enc.endEncoding();
        if crate::timing::labels() {
            command.setLabel(Some(&NSString::from_str(&format!(
                "last dispatch #{} {:#x} {groups:?}",
                self.dispatches, program.address
            ))));
        }
        let every = diag::config().gpu.commit_every;
        if every > 0 && (self.dispatches + 1).is_multiple_of(every) {
            self.commit();
        }
        if bound.is_some() {
            Ok(())
        } else {
            Err(DispatchError::Skipped("resources"))
        }
    }
}

/// How a compute program's grid is given.
#[derive(Clone, Copy)]
pub(super) enum Grid {
    Direct(crate::Dispatch),
    /// Three dimensions in guest memory at `at`, counting threads with `threads`.
    Indirect {
        at: u64,
        threads: bool,
    },
}

/// Why a compute dispatch did not encode on the GPU.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DispatchError {
    /// Indirect dimensions must be read on the CPU by the command processor.
    FallbackCpu,
    Skipped(&'static str),
}

/// Turns an indirect dispatch's dimensions into Metal's threadgroup counts,
/// then the grid size in threads: none past the command processor's limit.
const INDIRECT_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;
kernel void indirect_sizes(device const uint* dims [[buffer(0)]], device uint* out [[buffer(1)]], constant uint4& p [[buffer(2)]]) {
    uint3 size = max(p.xyz, uint3(1u));
    uint3 n = uint3(dims[0], dims[1], dims[2]);
    uint3 g = p.w != 0u ? n / size + select(uint3(0u), uint3(1u), n % size != 0u) : n;
    ulong total = ulong(g.x) * ulong(g.y) * ulong(g.z);
    if (total > LIMITul) g = uint3(0u);
    out[0] = g.x; out[1] = g.y; out[2] = g.z; out[3] = 0u;
    out[4] = n.x; out[5] = n.y; out[6] = n.z; out[7] = 0u;
}
"#;

/// Dispatches that can hold a slot of loop guard words at once; a slot is
/// zeroed when handed out again.
pub(super) const GUARD_SLOTS: usize = 1 << 16;
