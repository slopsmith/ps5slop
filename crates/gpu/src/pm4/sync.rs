//! Waits, releases, queued writes, atomics, semaphores and rewind.

use super::*;

/// [`WAIT_LIMIT`] from now, not counting time a command processor spends
/// waiting for a compile: the queue that would end the wait may be that one.
pub(super) struct Deadline(Instant);

impl Deadline {
    pub(super) fn start() -> Self {
        Self(Instant::now())
    }

    pub(super) fn passed(&mut self) -> bool {
        if crate::compile::blocked() {
            self.0 = Instant::now();
        }
        self.0.elapsed() > WAIT_LIMIT
    }
}

/// Tells the host the command processor polls memory, for as long as it lives.
pub(super) struct Polling<'a>(&'a dyn Host);

impl<'a> Polling<'a> {
    pub(super) fn start(host: &'a dyn Host) -> Self {
        host.waiting(true);
        Self(host)
    }
}

impl Drop for Polling<'_> {
    fn drop(&mut self) {
        self.0.waiting(false);
    }
}

/// WAIT_REG_MEM's comparison `function` of the masked value with `reference`.
fn compare(function: u32, value: u64, reference: u64) -> bool {
    match function & 7 {
        0 => true,
        1 => value < reference,
        2 => value <= reference,
        3 => value == reference,
        4 => value != reference,
        5 => value >= reference,
        6 => value > reference,
        _ => false,
    }
}

pub(super) fn gpu_clock() -> u64 {
    savestate::clock::unix_nanos()
}

/// ATOMIC_MEM's operation `op` (the TC op numbers SharpEmu-mac uses) on
/// `current`: whether it applied (a compare-swap may not) and the new value.
fn atomic_op(op: u32, current: u64, source: u64, compare: u64) -> (bool, u64) {
    let v = match op {
        7 => source,
        8 if current == compare => source,
        8 => return (false, current),
        15 => current.wrapping_add(source),
        16 => current.wrapping_sub(source),
        17 => (current as i64).min(source as i64) as u64,
        18 => current.min(source),
        19 => (current as i64).max(source as i64) as u64,
        20 => current.max(source),
        21 => current & source,
        22 => current | source,
        23 => current ^ source,
        24 if current >= source => 0,
        24 => current + 1,
        25 if current == 0 || current > source => source,
        25 => current - 1,
        _ => current,
    };
    (true, v)
}

/// Applies `f` to the 32- or 64-bit value at `address` atomically where the
/// host maps it, and returns what `f` said about it.
fn atomic(
    host: &dyn Host,
    address: u64,
    wide: bool,
    f: impl Fn(u64) -> (bool, u64),
) -> Option<bool> {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::SeqCst};
    let size = if wide { 8 } else { 4 };
    let mut probe = [0u8; 8];
    if !address.is_multiple_of(size) || !host.read(address, &mut probe[..size as usize]) {
        return None;
    }
    let mut result = true;
    match host.host_pointer(address) {
        // SAFETY: the address is aligned and readable guest memory, which the
        // host maps for the life of the process.
        Some(ptr) if wide => {
            let cell = unsafe { AtomicU64::from_ptr(ptr.cast()) };
            let _ = cell.try_update(SeqCst, SeqCst, |v| {
                let (ok, new) = f(v);
                result = ok;
                Some(new)
            });
        }
        Some(ptr) => {
            let cell = unsafe { AtomicU32::from_ptr(ptr.cast()) };
            let _ = cell.try_update(SeqCst, SeqCst, |v| {
                let (ok, new) = f(v as u64);
                result = ok;
                Some(new as u32)
            });
        }
        None => {
            let (ok, new) = f(if wide {
                u64::from_le_bytes(probe)
            } else {
                u32::from_le_bytes(probe[..4].try_into().unwrap()) as u64
            });
            result = ok;
            host.write(address, &new.to_le_bytes()[..size as usize]);
        }
    }
    Some(result)
}

impl Processor {
    /// ATOMIC_MEM: one read-modify-write after the work before it, or with
    /// command 1 a compare-swap retried until it succeeds.
    pub(super) fn atomic_mem(&mut self, host: &dyn Host, p: &[u32]) {
        let op = p[0] & 0x1F;
        let wide = p[0] & 0x20 != 0;
        let command = (p[0] >> 8) & 0xF;
        let (address, source, compare) = (addr(p[1], p[2]), addr(p[3], p[4]), addr(p[5], p[6]));
        self.wait_gpu(host, "ATOMIC_MEM");
        let mut deadline = Deadline::start();
        let mut polling = None;
        loop {
            let passed = atomic(host, address, wide, |v| atomic_op(op, v, source, compare));
            if command != 1 || op != 8 || passed != Some(false) || host.exiting() {
                return;
            }
            polling.get_or_insert_with(|| Polling::start(host));
            if deadline.passed() {
                self.stats.timeouts += 1;
                return;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// MEM_SEMAPHORE: select 6 signals (adds one to the 64-bit count, or sets
    /// it to 1), select 7 waits until the count is not zero and takes one.
    pub(super) fn mem_semaphore(&mut self, host: &dyn Host, p: &[u32]) {
        let address = (p[0] & !7) as u64 | (p[1] as u64) << 32;
        let select = p[2] >> 29;
        match select {
            6 => {
                self.wait_gpu(host, "MEM_SEMAPHORE signal");
                let set = p[2] & (1 << 20) != 0;
                atomic(host, address, true, |v| {
                    (true, if set { 1 } else { v.wrapping_add(1) })
                });
            }
            7 => {
                self.drain(host);
                let mut deadline = Deadline::start();
                let mut polling = None;
                while atomic(host, address, true, |v| (v != 0, v.saturating_sub(1))) == Some(false)
                {
                    polling.get_or_insert_with(|| Polling::start(host));
                    if host.exiting() {
                        return;
                    }
                    if deadline.passed() {
                        self.stats.timeouts += 1;
                        tracing::warn!("semaphore wait at {address:#x} timed out");
                        return;
                    }
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
            _ => {}
        }
    }

    /// REWIND: waits until the CPU sets the valid bit at `body`; the caller then
    /// fetches the rest of the buffer again, which the CPU may have written since.
    pub(super) fn rewind(&mut self, host: &dyn Host, body: u64) {
        let valid = |host: &dyn Host| {
            let mut b = [0u8; 4];
            host.read(body, &mut b);
            u32::from_le_bytes(b) & REWIND_VALID != 0
        };
        if valid(host) {
            return;
        }
        self.drain(host);
        trace(format_args!("rewind {:#x} waits", body - 4));
        let start = Instant::now();
        let mut warned = false;
        let _polling = Polling::start(host);
        while !valid(host) && !host.exiting() {
            if !warned && start.elapsed() > WAIT_LIMIT {
                warned = true;
                tracing::warn!(
                    "REWIND at {:#x} still not valid after {} s",
                    body - 4,
                    WAIT_LIMIT.as_secs()
                );
            }
            let t = crate::timing::start();
            std::thread::sleep(Duration::from_micros(50));
            crate::timing::add(crate::timing::Part::WaitMemory, t);
        }
    }

    /// Waits for the GPU work so far and makes pending releases take effect,
    /// before `why` reads what that work may have written.
    pub(super) fn wait_gpu(&mut self, host: &dyn Host, why: &'static str) {
        *self.stats.gpu_waits.entry(why).or_default() += 1;
        self.drain_now(host);
        self.sync();
    }

    pub(super) fn sync(&mut self) {
        if let Some(b) = self.backend.as_mut() {
            let t = crate::timing::start();
            b.sync();
            crate::timing::add(crate::timing::Part::Sync, t);
        }
    }

    pub(super) fn write_values(&mut self, host: &dyn Host, dest: u64, values: &[u32], fixed: bool) {
        trace(format_args!(
            "write {dest:#x} {:x?}",
            &values[..values.len().min(4)]
        ));
        if let Some(b) = self.backend.as_mut()
            && !diag::config().gpu.sync_releases
        {
            let write = crate::MemoryWrite {
                dest,
                values,
                fixed,
            };
            // Only a queue that never draws: draws read guest memory through
            // other buffer objects, which Metal does not order against the
            // write. Queued entries are applied later, so a write that lands
            // now would end up before them.
            let harmless =
                |p: &Pending| matches!(p, Pending::Write { landed: true, .. } | Pending::Interrupt);
            let landed = self.stats.draws == 0
                && self.pending.iter().all(harmless)
                && self.retiring.iter().all(|(_, l)| l.iter().all(harmless))
                && b.write_memory(host, &write);
            self.pending.push(Pending::Write {
                dest,
                values: values.to_vec(),
                fixed,
                landed,
            });
            return;
        }
        Self::apply_write(host, dest, values, fixed);
    }

    fn apply_write(host: &dyn Host, dest: u64, values: &[u32], fixed: bool) {
        for (i, v) in values.iter().enumerate() {
            let at = if fixed { dest } else { dest + 4 * i as u64 };
            host.write(at, &v.to_le_bytes());
        }
    }

    /// An end-of-pipe release: data select 1 writes 32 bits, 2 writes 64,
    /// 3 and 4 a clock; interrupt selects 1 and 2 raise an interrupt.
    // Selectors 5 and 6 write nothing and interrupt only while the label at
    // `dest` is at or below `value` (SharpEmu-mac's `InterruptDecision`).
    pub(super) fn release(
        &mut self,
        host: &dyn Host,
        data: u32,
        interrupt: u32,
        dest: u64,
        value: u64,
        context: u32,
    ) {
        trace(format_args!(
            "release data {data} interrupt {interrupt} {dest:#x} = {value:#x}"
        ));
        if self.backend.is_some() && !diag::config().gpu.sync_releases {
            self.pending.push(Pending::Release {
                data,
                interrupt,
                dest,
                value,
                context,
            });
        } else {
            self.sync();
            self.apply_release(host, data, interrupt, dest, value, context);
        }
    }

    /// Ends a submission: pending releases take effect, and the GPU work
    /// encoded so far is done.
    pub fn finish(&mut self, host: &dyn Host) {
        self.drain(host);
        self.sync();
    }

    /// Ends a submission and raises the queue's interrupt. One that ends a
    /// frame waits for the GPU ([`Processor::finish`]); for others the work is
    /// committed and its pending releases, then the interrupt, take effect once
    /// the GPU has done it ([`Processor::retire`]), without waiting here.
    pub fn end_submission(&mut self, host: &dyn Host, ends_frame: bool) {
        let queued = !self.pending.is_empty() || !self.retiring.is_empty();
        if !ends_frame
            && queued
            && !diag::config().gpu.sync_releases
            && !diag::config().gpu.wait_submissions
            && let Some(b) = self.backend.as_mut()
        {
            let point = b.commit_point();
            self.pending.push(Pending::Interrupt);
            self.retiring
                .push_back((point, std::mem::take(&mut self.pending)));
            self.retire(host);
            return;
        }
        if ends_frame {
            self.finish(host);
        } else {
            self.drain(host);
        }
        host.interrupt(0);
    }

    /// Makes the pending work of ended submissions whose GPU work is done take
    /// effect, in order. Returns whether some is still waiting for the GPU.
    pub fn retire(&mut self, host: &dyn Host) -> bool {
        while let Some(&(point, _)) = self.retiring.front() {
            if !self.backend.as_mut().is_none_or(|b| b.done(point)) {
                break;
            }
            let (_, list) = self.retiring.pop_front().expect("checked above");
            for p in list {
                self.apply(host, p);
            }
        }
        !self.retiring.is_empty()
    }

    /// Waits for the GPU work behind pending releases, then makes them take
    /// effect in order. Called before anything that could observe them.
    pub fn drain(&mut self, host: &dyn Host) {
        if self.pending.is_empty() && self.retiring.is_empty() {
            return;
        }
        *self
            .stats
            .gpu_waits
            .entry("releases before a poll or flip")
            .or_default() += 1;
        self.drain_now(host);
    }

    fn drain_now(&mut self, host: &dyn Host) {
        self.sync();
        for (_, list) in std::mem::take(&mut self.retiring) {
            for p in list {
                self.apply(host, p);
            }
        }
        for p in std::mem::take(&mut self.pending) {
            self.apply(host, p);
        }
    }

    fn apply(&mut self, host: &dyn Host, p: Pending) {
        match p {
            Pending::Release {
                data,
                interrupt,
                dest,
                value,
                context,
            } => self.apply_release(host, data, interrupt, dest, value, context),
            Pending::Write {
                dest,
                values,
                fixed,
                landed: false,
            } => Self::apply_write(host, dest, &values, fixed),
            Pending::Write { .. } => {}
            Pending::Interrupt => host.interrupt(0),
        }
    }

    /// The 32-bit value queued work will leave at `address`, the latest first.
    fn queued_word(&self, address: u64) -> Option<u32> {
        self.pending
            .iter()
            .rev()
            .chain(self.retiring.iter().rev().flat_map(|(_, l)| l.iter().rev()))
            .find_map(|p| p.word_at(address))
    }

    fn apply_release(
        &mut self,
        host: &dyn Host,
        data: u32,
        interrupt: u32,
        dest: u64,
        value: u64,
        context: u32,
    ) {
        let raise = match interrupt {
            1 | 2 | 4 => true,
            5 => dest != 0 && (read_u64(host, dest) as u32) <= value as u32,
            6 => dest != 0 && read_u64(host, dest) <= value,
            _ => false,
        };
        if interrupt <= 3 {
            match data {
                1 => {
                    host.write(dest, &(value as u32).to_le_bytes());
                }
                2 => {
                    host.write(dest, &value.to_le_bytes());
                }
                3 | 4 => {
                    host.write(dest, &gpu_clock().to_le_bytes());
                }
                7 => {
                    atomic(host, dest, false, |v| {
                        (true, v.wrapping_add(value & 0xFFFF_FFFF))
                    });
                }
                _ => {}
            }
        }
        if raise {
            self.stats.interrupts += 1;
            host.interrupt(context);
        }
    }

    /// Polls guest memory until the masked value passes the comparison in
    /// `control`, the process exits, or [`WAIT_LIMIT`] passes.
    pub(super) fn wait(
        &mut self,
        host: &dyn Host,
        control: u32,
        address: u64,
        reference: u64,
        mask: u64,
        wide: bool,
    ) {
        trace(format_args!(
            "wait {address:#x} & {mask:#x} function {} {reference:#x}",
            control & 7
        ));
        // GPU work runs in order, so queued releases and writes come first; with
        // none at the label, memory holds what the wait sees (unless a program writes it).
        let word = |at: u64| self.queued_word(at);
        let read = |host: &dyn Host| {
            if wide {
                read_u64(host, address)
            } else {
                let mut b = [0u8; 4];
                host.read(address, &mut b);
                u32::from_le_bytes(b) as u64
            }
        };
        let value = match (word(address), wide.then(|| word(address + 4))) {
            (Some(lo), None) => Some(lo as u64),
            (Some(lo), Some(Some(hi))) => Some(lo as u64 | (hi as u64) << 32),
            (None, None | Some(None)) => Some(read(host)),
            _ => None,
        };
        if value.is_some_and(|v| compare(control, v & mask, reference)) {
            if word(address).is_some() {
                self.stats.deferred_waits += 1;
            }
            return;
        }
        self.drain(host);
        let mut deadline = Deadline::start();
        let mut polling = None;
        loop {
            let value = read(host);
            if compare(control, value & mask, reference) || host.exiting() {
                return;
            }
            if deadline.passed() {
                self.stats.timeouts += 1;
                if self.stats.timeouts <= 16 {
                    tracing::warn!(
                        "wait timed out on {}: {address:#x} & {mask:#x} = {value:#x}, function {}, reference {reference:#x}",
                        std::thread::current().name().unwrap_or("?"),
                        control & 7
                    );
                }
                return;
            }
            polling.get_or_insert_with(|| Polling::start(host));
            let t = crate::timing::start();
            std::thread::sleep(Duration::from_micros(200));
            crate::timing::add(crate::timing::Part::WaitMemory, t);
        }
    }
}
