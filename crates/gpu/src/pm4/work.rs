//! Draws, dispatches, flips and NOP markers.

use super::*;

impl Processor {
    /// `n` argument dwords of an indirect draw or dispatch at `offset` past
    /// the indirect base, read once earlier GPU work (which may write them) is done.
    pub(super) fn indirect_args(&mut self, host: &dyn Host, offset: u32, n: usize) -> Vec<u32> {
        self.wait_gpu(host, "indirect arguments");
        read_dwords(host, self.indirect_base + offset as u64, n).unwrap_or_else(|_| vec![0; n])
    }

    pub(super) fn dispatch(&mut self, host: &dyn Host, dims: [u32; 3], initiator: u32) {
        self.stats.dispatches += 1;
        let dispatch = self.groups(dims, initiator);
        let groups = dispatch.groups;
        let total = groups.iter().map(|&g| g as u64).product::<u64>();
        let program = (self.sh[crate::regs::sh::COMPUTE_PGM_LO] as u64
            | (self.sh[crate::regs::sh::COMPUTE_PGM_LO + 1] as u64) << 32)
            << 8;
        trace(format_args!("dispatch {groups:?} program {program:#x}"));
        if diag::config().gpu.probe == Some(program) {
            let user = &self.sh
                [crate::regs::sh::COMPUTE_USER_DATA_0..crate::regs::sh::COMPUTE_USER_DATA_0 + 16];
            tracing::info!(
                target: "gpu::probe",
                "[{}]: dispatch {groups:?} of {program:#x}, user data {user:x?}",
                std::thread::current().name().unwrap_or("?")
            );
        }
        if total > MAX_GROUPS {
            self.stats.huge_dispatches += 1;
            diag::problems::record_with_detail(
                diag::problems::Kind::DispatchSkipped,
                "too many groups",
                format!("cs {program:#x}"),
                format!("{groups:?}"),
            );
            if self.stats.huge_dispatches <= 16 {
                tracing::error!(
                    "dispatch of {groups:?} groups of the program at {program:#x} skipped"
                );
            }
            return;
        }
        if let Some(b) = self.backend.as_mut() {
            let state = State {
                context: &self.context,
                sh: &self.sh,
                uconfig: &self.uconfig,
                color_control_written: self.color_control_written,
            };
            let t = crate::timing::start();
            b.dispatch(host, &state, dispatch);
            crate::timing::add(crate::timing::Part::Dispatch, t);
        }
        if diag::config().gpu.sync_dispatches {
            self.sync();
            trace(format_args!("dispatch done"));
        }
        self.gpu_watch(host, || format!("dispatch {groups:?} of {program:#x}"));
    }

    /// A dispatch whose dimensions are at `at`: the backend reads them on the
    /// GPU when it can, or they are read here once earlier GPU work is done.
    pub(super) fn dispatch_indirect(&mut self, host: &dyn Host, at: u64, initiator: u32) {
        let program = (self.sh[crate::regs::sh::COMPUTE_PGM_LO] as u64
            | (self.sh[crate::regs::sh::COMPUTE_PGM_LO + 1] as u64) << 32)
            << 8;
        let config = &diag::config().gpu;
        let debugging =
            config.sync_dispatches || config.probe == Some(program) || config.watch.is_some();
        if !debugging
            && !diag::config().gpu.cpu_indirect
            && let Some(b) = self.backend.as_mut()
        {
            let state = State {
                context: &self.context,
                sh: &self.sh,
                uconfig: &self.uconfig,
                color_control_written: self.color_control_written,
            };
            let t = crate::timing::start();
            let done = b.dispatch_indirect(host, &state, at, initiator & 0x20 != 0);
            crate::timing::add(crate::timing::Part::Dispatch, t);
            if done {
                self.stats.dispatches += 1;
                self.stats.gpu_indirect += 1;
                trace(format_args!(
                    "dispatch with dimensions at {at:#x} on the GPU, program {program:#x}"
                ));
                return;
            }
        }
        self.wait_gpu(host, "indirect arguments");
        let a = read_dwords(host, at, 3).unwrap_or_else(|_| vec![0; 3]);
        self.dispatch(host, [a[0], a[1], a[2]], initiator);
    }

    /// With USE_THREAD_DIMENSIONS (initiator bit 5) the dimensions count
    /// threads, and a last group they do not fill runs partly (as radv uses it).
    fn groups(&self, dims: [u32; 3], initiator: u32) -> crate::Dispatch {
        if initiator & 0x20 == 0 {
            return crate::Dispatch::groups(dims);
        }
        let size =
            [0, 1, 2].map(|i| (self.sh[crate::regs::sh::COMPUTE_NUM_THREAD_X + i] & 0xFFFF).max(1));
        let groups = [0, 1, 2].map(|i| dims[i].div_ceil(size[i]));
        let partial = (0..3).any(|i| !dims[i].is_multiple_of(size[i]));
        crate::Dispatch {
            groups,
            threads: partial.then_some(dims),
        }
    }

    /// With a watched word, waits for each draw and dispatch and reports
    /// those that change it.
    fn gpu_watch(&mut self, host: &dyn Host, what: impl FnOnce() -> String) {
        static AT: std::sync::OnceLock<Option<(u64, Instant)>> = std::sync::OnceLock::new();
        let Some((at, from)) = *AT.get_or_init(|| {
            let (at, delay) = diag::config().gpu.watch?;
            Some((at, Instant::now() + delay))
        }) else {
            return;
        };
        if Instant::now() < from {
            return;
        }
        self.sync();
        let now = read_u64(host, at);
        if self.watched != Some(now) {
            if self.watched.is_some() {
                let user = &self.sh[crate::regs::sh::COMPUTE_USER_DATA_0
                    ..crate::regs::sh::COMPUTE_USER_DATA_0 + 16];
                tracing::info!(
                    target: "gpu::watch",
                    "{:.3}: {at:#x} {:#x?} -> {now:#x} after {} on {} (compute user {user:x?})",
                    crate::seconds(),
                    self.watched,
                    what(),
                    std::thread::current().name().unwrap_or("?")
                );
            }
            self.watched = Some(now);
        }
    }

    pub(super) fn draw(&mut self, host: &dyn Host, op: u32, draw: Draw) {
        self.stats.draws += 1;
        if let Some(b) = self.backend.as_mut() {
            let state = State {
                context: &self.context,
                sh: &self.sh,
                uconfig: &self.uconfig,
                color_control_written: self.color_control_written,
            };
            let t = crate::timing::start();
            b.draw(host, &state, draw);
            crate::timing::add(crate::timing::Part::Draw, t);
        }
        self.color_control_written = false;
        self.gpu_watch(host, || format!("draw {op:#x}"));
    }

    pub(super) fn nop(&mut self, host: &dyn Host, code: u32, p: &[u32]) {
        match code {
            0 if p.first().is_some_and(|w| w & 0xFFFF_0000 == MARKER_MAGIC) => self.marker(host, p),
            NOP_FLIP if p.len() >= 5 => self.flip(
                host,
                Flip {
                    handle: p[0] as i32,
                    index: p[1] as i32,
                    mode: p[2] as i32,
                    arg: addr(p[3], p[4]) as i64,
                },
            ),
            NOP_WRITE_DATA if p.len() >= 3 => {
                let control = p[0];
                if WRITE_DATA_MEMORY.contains(&(control & 0xFF)) {
                    let fixed = (control >> 16) & 0xFF != 0;
                    self.write_values(host, addr(p[1], p[2]), &p[3..], fixed);
                }
            }
            NOP_RELEASE_MEM if p.len() >= 7 => {
                let control = p[1];
                let interrupt = (control >> 24) & 0xFF;
                let data = (control >> 16) & 0xFF;
                self.release(
                    host,
                    data,
                    interrupt,
                    addr(p[2], p[3]),
                    addr(p[4], p[5]),
                    p[6] & 0x07FF_FFFF,
                );
            }
            NOP_WAIT_MEMORY_32 if p.len() >= 5 => {
                let (mask, reference, control) = if p.len() == 5 {
                    (p[2], p[4], p[3])
                } else {
                    (p[2], p[3], p[4])
                };
                self.wait(
                    host,
                    control,
                    addr(p[0], p[1]),
                    reference as u64,
                    mask as u64,
                    false,
                );
            }
            NOP_WAIT_MEMORY_64 if p.len() >= 7 => {
                self.wait(
                    host,
                    p[6],
                    addr(p[0], p[1]),
                    addr(p[4], p[5]),
                    addr(p[2], p[3]),
                    true,
                );
            }
            NOP_WAIT_FLIP_DONE => {}
            _ => {}
        }
    }

    fn marker(&mut self, host: &dyn Host, p: &[u32]) {
        let flip = Flip {
            handle: 0,
            index: -1,
            mode: 0,
            arg: 0,
        };
        match p[0] & 0xFFF {
            MARKER_FLIP => self.flip(host, flip),
            MARKER_FLIP_WITH_LABEL if p.len() >= 4 => {
                self.flip(host, flip);
                host.write(addr(p[1], p[2]), &p[3].to_le_bytes());
            }
            MARKER_FLIP_WITH_INTERRUPT if p.len() >= 6 => {
                self.flip(host, flip);
                host.write(addr(p[1], p[2]), &p[3].to_le_bytes());
                self.stats.interrupts += 1;
                host.interrupt(p[5]);
            }
            _ => {}
        }
    }

    fn flip(&mut self, host: &dyn Host, flip: Flip) {
        self.drain(host);
        if let Some(b) = self.backend.as_mut() {
            b.frame_done(host.display_buffer(flip.index));
        }
        if let Some(last) = self.last_flip
            && let Some(wait) = (last + VBLANK).checked_duration_since(Instant::now())
        {
            let t = crate::timing::start();
            std::thread::sleep(wait);
            crate::timing::add(crate::timing::Part::Pacing, t);
        }
        let now = Instant::now();
        if let Some(last) = self.last_flip {
            let gap = now - last;
            self.stats.longest_flip_gap = self.stats.longest_flip_gap.max(gap);
            if gap > Duration::from_secs(1) && self.stats.long_flip_gaps.len() < 64 {
                self.stats.long_flip_gaps.push(gap);
            }
        }
        self.last_flip = Some(now);
        self.stats.flips += 1;
        host.flip(flip);
    }
}
