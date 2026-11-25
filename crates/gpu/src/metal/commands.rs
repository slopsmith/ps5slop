//! Command buffers: opening, committing, waiting, and reporting failures.

use super::*;

impl Metal {
    pub(super) fn command(&mut self) -> Retained<CommandBuffer> {
        if self.command.is_none() {
            self.command = self.new_command_buffer();
        }
        self.command
            .clone()
            .expect("Metal hands out command buffers")
    }

    /// A command buffer; with `PS5SLOP_GPU_ENCODER_STATUS`, one whose errors say which encoder failed.
    fn new_command_buffer(&self) -> Option<Retained<CommandBuffer>> {
        if !diag::config().gpu.encoder_status {
            return self.queue.commandBuffer();
        }
        let d = objc2_metal::MTLCommandBufferDescriptor::new();
        d.setErrorOptions(objc2_metal::MTLCommandBufferErrorOption::EncoderExecutionStatus);
        self.queue.commandBufferWithDescriptor(&d)
    }

    /// The command buffer for a texture upload the next draw needs.
    pub(super) fn upload_command(&mut self) -> Retained<CommandBuffer> {
        if self.pending_writes || diag::config().gpu.inline_uploads {
            self.end_pass();
            return self.command();
        }
        if self.upload.is_none() {
            self.upload = self.new_command_buffer();
        }
        self.upload
            .clone()
            .expect("Metal hands out command buffers")
    }

    /// The fence encoders that read guest memory wait for, once a writer updated it:
    /// Metal orders work only within one buffer, and our windows are several.
    pub(super) fn wait_memory(&self) -> Option<&ProtocolObject<dyn MTLFence>> {
        self.fence.as_deref().filter(|_| self.fence_updated)
    }

    #[track_caller]
    pub(super) fn end_pass(&mut self) {
        if let Some((enc, addresses, colors, depth)) = self.pass.take() {
            *self
                .pass_ends
                .entry(std::panic::Location::caller().line())
                .or_default() += 1;
            if let Some(f) = self.fence.as_deref()
                && std::mem::take(&mut self.pass_memory)
            {
                enc.updateFence_afterStages(f, MTLRenderStages::Fragment);
                self.fence_updated = true;
            }
            enc.endEncoding();
            if self.snap.is_some() {
                self.snap_pass(&addresses, &colors, depth);
            }
        }
    }

    pub(super) fn commit(&mut self) {
        self.end_pass();
        self.stale = 0;
        let now = crate::timing::host_seconds();
        self.commits += 1;
        self.uncommitted_draws = 0;
        if let Some(cb) = self.upload.take() {
            cb.commit();
            self.inflight.push((cb, now, self.commits));
        }
        self.pending_writes = false;
        if let Some(cb) = self.command.take() {
            cb.commit();
            self.inflight.push((cb, now, self.commits));
        }
        if self.inflight.len() > 32 {
            self.retire_completed();
            if self.inflight.len() > 48 {
                wait_reporting_hangs(&self.inflight[0].0);
            }
            if let Some((cb, at, _)) = self.inflight.first()
                && now - at > 2.0
                && self.reported_stuck != Some(Retained::as_ptr(cb) as usize)
            {
                self.reported_stuck = Some(Retained::as_ptr(cb) as usize);
                tracing::warn!(
                    "command buffer {:?} still running after 2 s (status {:?})",
                    cb.label().map(|l| l.to_string()),
                    cb.status()
                );
            }
        }
    }

    pub(super) fn flush_and_wait(&mut self) {
        self.commit();
        self.written = Written::Nothing;
        self.gpu_checked.clear();
        for (cb, committed, _) in std::mem::take(&mut self.inflight) {
            wait_reporting_hangs(&cb);
            self.account(&cb, committed, true);
        }
        self.check_guards();
    }

    /// Drops the command buffers that have finished from `inflight`.
    pub(super) fn retire_completed(&mut self) {
        let mut i = 0;
        while i < self.inflight.len() {
            if matches!(
                self.inflight[i].0.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            ) {
                let (cb, committed, _) = self.inflight.remove(i);
                self.account(&cb, committed, false);
            } else {
                i += 1;
            }
        }
    }

    /// Adds a finished command buffer's times to the profile and reports its
    /// error; `waited` when the GPU thread waited for it.
    fn account(&mut self, cb: &CommandBuffer, committed: f64, waited: bool) {
        let ns = |s: f64| (s.max(0.0) * 1e9) as u64;
        crate::timing::add_raw(
            crate::timing::Part::Submission,
            ns(cb.kernelStartTime() - committed),
        );
        if waited && cb.GPUEndTime() > 0.0 {
            let woken = crate::timing::host_seconds();
            crate::timing::add_raw(crate::timing::Part::WakeUp, ns(woken - cb.GPUEndTime()));
        }
        if let Some(e) = cb.error() {
            self.failed_commands += 1;
            if self.failed_commands <= 8 {
                let windows: u64 = self.windows.iter().map(|w| w.end - w.start).sum();
                tracing::error!(
                    "command buffer failed: {} (code {}); {} in flight, windows {} MiB + area {} MiB, Metal allocated {} MiB of {} MiB",
                    e.localizedDescription(),
                    e.code(),
                    self.inflight.len(),
                    windows >> 20,
                    [&self.area, &self.area_high]
                        .iter()
                        .map(|a| a.as_ref().map_or(0, |w| w.end - w.start))
                        .sum::<u64>()
                        >> 20,
                    self.device.currentAllocatedSize() >> 20,
                    self.device.recommendedMaxWorkingSetSize() >> 20,
                );
                report_encoders(&e);
            }
        }
        if cb.GPUStartTime() > 0.0 && cb.kernelStartTime() > 0.0 {
            let busy = (cb.GPUEndTime() - cb.GPUStartTime()).max(0.0);
            self.gpu_time_accum += busy;
            let scheduled = cb.GPUStartTime() - cb.kernelStartTime();
            crate::timing::add_raw(
                crate::timing::Part::Scheduling,
                (scheduled.max(0.0) * 1e9) as u64,
            );
            crate::timing::add_raw(crate::timing::Part::GpuBusy, (busy * 1e9) as u64);
        }
        if waited {
            crate::timing::add_raw(crate::timing::Part::Syncs, 1_000_000);
        }
    }
}

/// Waits for `cb`; one still running after 10 s is reported with its label
/// (`PS5SLOP_GPU_LABELS` names the dispatches in it), then waited for.
fn wait_reporting_hangs(cb: &CommandBuffer) {
    let t = crate::timing::start();
    wait_for(cb);
    crate::timing::add(crate::timing::Part::CommandWait, t);
}

fn wait_for(cb: &CommandBuffer) {
    let start = std::time::Instant::now();
    while !matches!(
        cb.status(),
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    ) {
        if start.elapsed() > std::time::Duration::from_secs(10) {
            tracing::warn!(
                "command buffer {:?} still running after 10 s (status {:?})",
                cb.label().map(|l| l.to_string()),
                cb.status()
            );
            cb.waitUntilCompleted();
            return;
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
}

/// Logs the encoders a failed command buffer reports as faulted or pending
/// (`PS5SLOP_GPU_ENCODER_STATUS`), with their labels and signposts.
fn report_encoders(e: &objc2_foundation::NSError) {
    use objc2_metal::{
        MTLCommandBufferEncoderInfo, MTLCommandBufferEncoderInfoErrorKey,
        MTLCommandEncoderErrorState as State,
    };
    let info = e.userInfo();
    // SAFETY: Metal documents this key's value as an array of MTLCommandBufferEncoderInfo.
    let Some(list) = (unsafe { info.objectForKey(MTLCommandBufferEncoderInfoErrorKey) }) else {
        return;
    };
    let list: Retained<objc2_foundation::NSArray<ProtocolObject<dyn MTLCommandBufferEncoderInfo>>> =
        unsafe { Retained::cast_unchecked(list) };
    for enc in list.iter() {
        let state = enc.errorState();
        if state == State::Faulted || state == State::Pending {
            // SAFETY: `debugSignposts` returns an array of strings, or nil when there are none.
            let posts: Option<Retained<objc2_foundation::NSArray<NSString>>> =
                unsafe { objc2::msg_send![&*enc, debugSignposts] };
            let posts: Vec<String> = posts
                .map(|p| p.iter().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            let n = posts.len();
            tracing::error!(
                "  encoder {:?} {} {:?}",
                state.0,
                enc.label(),
                &posts[n.saturating_sub(4)..]
            );
        }
    }
}
