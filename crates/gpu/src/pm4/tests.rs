use std::cell::RefCell;

use super::*;

#[derive(Default)]
struct Fake {
    mem: RefCell<Vec<u8>>,
    interrupts: RefCell<Vec<u32>>,
    flips: RefCell<Vec<Flip>>,
}

impl Host for Fake {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        let mem = self.mem.borrow();
        let Some(src) = mem.get(addr as usize..addr as usize + buf.len()) else {
            return false;
        };
        buf.copy_from_slice(src);
        true
    }
    fn write(&self, addr: u64, data: &[u8]) -> bool {
        let mut mem = self.mem.borrow_mut();
        let Some(dst) = mem.get_mut(addr as usize..addr as usize + data.len()) else {
            return false;
        };
        dst.copy_from_slice(data);
        true
    }
    fn interrupt(&self, context: u32) {
        self.interrupts.borrow_mut().push(context);
    }
    fn flip(&self, flip: Flip) {
        self.flips.borrow_mut().push(flip);
    }
    fn exiting(&self) -> bool {
        false
    }
}

fn header(len: u32, op: u32, code: u32) -> u32 {
    0xC000_0000 | ((len - 2) << 16) | (op << 8) | (code << 2)
}

fn run(packets: &[u32]) -> (Fake, Processor) {
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    host.write(0x100, &bytes);
    let mut p = Processor::default();
    p.run(&host, 0x100, packets.len() as u32).unwrap();
    (host, p)
}

fn word(host: &Fake, at: usize) -> u32 {
    u32::from_le_bytes(host.mem.borrow()[at..at + 4].try_into().unwrap())
}

#[test]
fn dma_data_fills_the_gds_and_copies_it_to_memory() {
    let (host, _) = run(&[
        // Data 0x2A into GDS bytes 8..12.
        header(7, IT_DMA_DATA, 0),
        2 << 29 | 1 << 20,
        0x2A,
        0,
        8,
        0,
        4,
        // GDS bytes 8..12 to memory at 0x900.
        header(7, IT_DMA_DATA, 0),
        1 << 29 | 3 << 20,
        8,
        0,
        0x900,
        0,
        4,
    ]);
    assert_eq!(word(&host, 0x900), 0x2A);
}

#[test]
fn indirect_dispatches_read_their_group_counts_past_the_base() {
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    host.write(0xA08, &[2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0]);
    let packets = [
        header(4, IT_SET_BASE, 0),
        1,
        0xA00,
        0,
        header(3, IT_DISPATCH_INDIRECT, 0),
        8,
        0,
    ];
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    host.write(0x100, &bytes);
    struct Groups(std::sync::Arc<std::sync::Mutex<Vec<[u32; 3]>>>);
    impl Backend for Groups {
        fn draw(&mut self, _: &dyn Host, _: &State, _: Draw) {}
        fn dispatch(&mut self, _: &dyn Host, _: &State, d: crate::Dispatch) {
            self.0.lock().unwrap().push(d.groups);
        }
        fn sync(&mut self) {}
        fn frame_done(&mut self, _: Option<u64>) {}
    }
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut p = Processor {
        backend: Some(Box::new(Groups(seen.clone()))),
        ..Processor::default()
    };
    p.run(&host, 0x100, packets.len() as u32).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![[2, 3, 1]]);
}

struct Dispatches(std::sync::Arc<std::sync::Mutex<Vec<crate::Dispatch>>>);
impl Backend for Dispatches {
    fn draw(&mut self, _: &dyn Host, _: &State, _: Draw) {}
    fn dispatch(&mut self, _: &dyn Host, _: &State, d: crate::Dispatch) {
        self.0.lock().unwrap().push(d);
    }
    fn sync(&mut self) {}
    fn frame_done(&mut self, _: Option<u64>) {}
}

fn dispatches(host: &Fake, packets: &[u32]) -> Vec<crate::Dispatch> {
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    host.write(0x100, &bytes);
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut p = Processor {
        backend: Some(Box::new(Dispatches(seen.clone()))),
        ..Processor::default()
    };
    p.run(host, 0x100, packets.len() as u32).unwrap();
    seen.lock().unwrap().clone()
}

#[test]
fn dispatches_in_thread_dimensions_run_whole_and_partial_groups() {
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    let num_thread = crate::regs::sh::COMPUTE_NUM_THREAD_X as u32;
    let packets = [
        header(5, IT_SET_SH_REG, 0),
        num_thread,
        64,
        1,
        1,
        header(5, IT_DISPATCH_DIRECT, 0),
        18432,
        1,
        1,
        0x61,
        header(5, IT_DISPATCH_DIRECT, 0),
        100,
        1,
        1,
        0x61,
        header(5, IT_DISPATCH_DIRECT, 0),
        3,
        1,
        1,
        0x41,
    ];
    let seen = dispatches(&host, &packets);
    let d = |groups, threads| crate::Dispatch { groups, threads };
    assert_eq!(
        seen,
        vec![
            d([288, 1, 1], None),
            d([2, 1, 1], Some([100, 1, 1])),
            d([3, 1, 1], None),
        ]
    );
}

#[test]
fn compute_queue_indirect_dispatches_read_a_full_address() {
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    host.write(0xA08, &[2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0]);
    let packets = [header(4, IT_DISPATCH_INDIRECT, 0), 0xA08, 0, 0x41];
    let seen = dispatches(&host, &packets);
    assert_eq!(
        seen.iter().map(|d| d.groups).collect::<Vec<_>>(),
        vec![[2, 3, 1]]
    );
}

#[test]
fn label_waits_pass_without_the_gpu_and_releases_wait_for_it() {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
    #[derive(Clone, Default)]
    struct Gpu {
        syncs: std::sync::Arc<AtomicU32>,
        done: std::sync::Arc<AtomicBool>,
    }
    impl Backend for Gpu {
        fn draw(&mut self, _: &dyn Host, _: &State, _: Draw) {}
        fn dispatch(&mut self, _: &dyn Host, _: &State, _: crate::Dispatch) {}
        fn sync(&mut self) {
            self.syncs.fetch_add(1, Relaxed);
            self.done.store(true, Relaxed);
        }
        fn flush(&mut self) {}
        fn commit_point(&mut self) -> u64 {
            7
        }
        fn done(&mut self, point: u64) -> bool {
            assert_eq!(point, 7);
            self.done.load(Relaxed)
        }
        fn frame_done(&mut self, _: Option<u64>) {}
    }
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    host.write(0x800, &5u32.to_le_bytes());
    // AGC's barrier: zero a label, release 1 into it, wait for 1.
    let packets = [
        header(7, IT_DMA_DATA, 0),
        2 << 29,
        0,
        0x800,
        0,
        0,
        4,
        header(8, IT_RELEASE_MEM, 0),
        0,
        (1 << 29) | (2 << 24),
        0x800,
        0,
        1,
        0,
        3,
        header(7, IT_WAIT_REG_MEM, 0),
        0x13,
        0x800,
        0,
        1,
        0xFFFF_FFFF,
        0,
    ];
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    host.write(0x100, &bytes);
    let gpu = Gpu::default();
    let mut p = Processor {
        backend: Some(Box::new(gpu.clone())),
        ..Processor::default()
    };
    p.run(&host, 0x100, packets.len() as u32).unwrap();
    p.end_submission(&host, false);
    assert_eq!(gpu.syncs.load(Relaxed), 0);
    assert_eq!(
        word(&host, 0x800),
        5,
        "nothing takes effect before the GPU is done"
    );
    assert!(host.interrupts.borrow().is_empty());
    assert!(p.retire(&host));
    gpu.done.store(true, Relaxed);
    assert!(!p.retire(&host));
    assert_eq!(word(&host, 0x800), 1);
    assert_eq!(*host.interrupts.borrow(), [3, 0]);
    assert_eq!(gpu.syncs.load(Relaxed), 0);
}

#[test]
fn queued_writes_and_releases_report_what_they_leave_in_memory() {
    let w = Pending::Write {
        dest: 0x100,
        values: vec![1, 2, 3],
        fixed: false,
        landed: false,
    };
    assert_eq!(
        (w.word_at(0x104), w.word_at(0x10c), w.word_at(0x102)),
        (Some(2), None, None)
    );
    let f = Pending::Write {
        dest: 0x100,
        values: vec![1, 2, 3],
        fixed: true,
        landed: true,
    };
    assert_eq!((f.word_at(0x100), f.word_at(0x104)), (Some(3), None));
    let r = Pending::Release {
        data: 2,
        interrupt: 0,
        dest: 0x200,
        value: 0x5_0000_0007,
        context: 0,
    };
    assert_eq!((r.word_at(0x200), r.word_at(0x204)), (Some(7), Some(5)));
}

#[test]
fn write_data_and_release_mem_reach_memory_and_interrupts() {
    let (host, p) = run(&[
        header(5, IT_WRITE_DATA, 0),
        5 << 8,
        0x800,
        0,
        0xAABB_CCDD,
        header(8, IT_RELEASE_MEM, 0),
        0,
        (2 << 29) | (2 << 24),
        0x810,
        0,
        0x1122_3344,
        0x5566_7788,
        7,
    ]);
    assert_eq!(word(&host, 0x800), 0xAABB_CCDD);
    assert_eq!(word(&host, 0x810), 0x1122_3344);
    assert_eq!(word(&host, 0x814), 0x5566_7788);
    assert_eq!(*host.interrupts.borrow(), [7]);
    assert_eq!(p.stats.packets, 2);
}

#[test]
fn a_write_after_a_queued_release_to_the_same_label_stays_last() {
    // Writes the backend takes land at once, as the GPU would run them.
    struct Landing;
    impl Backend for Landing {
        fn draw(&mut self, _: &dyn Host, _: &State, _: Draw) {}
        fn dispatch(&mut self, _: &dyn Host, _: &State, _: crate::Dispatch) {}
        fn sync(&mut self) {}
        fn frame_done(&mut self, _: Option<u64>) {}
        fn write_memory(&mut self, host: &dyn Host, w: &crate::MemoryWrite) -> bool {
            for (i, v) in w.values.iter().enumerate() {
                host.write(w.dest + 4 * i as u64, &v.to_le_bytes());
            }
            true
        }
    }
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    let packets = [
        header(8, IT_RELEASE_MEM, 0),
        0,
        1 << 29,
        0x900,
        0,
        1,
        0,
        0,
        header(5, IT_WRITE_DATA, 0),
        5 << 8,
        0x900,
        0,
        0,
    ];
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    host.write(0x100, &bytes);
    let mut p = Processor {
        backend: Some(Box::new(Landing)),
        ..Processor::default()
    };
    p.run(&host, 0x100, packets.len() as u32).unwrap();
    p.finish(&host);
    assert_eq!(word(&host, 0x900), 0);
}

#[test]
fn registers_are_recorded_and_indirect_buffers_followed() {
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    let inner = [header(4, IT_SET_SH_REG, 0), 0x8, 0x1234, 0x5678];
    host.write(
        0x400,
        &inner
            .iter()
            .flat_map(|d| d.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let outer = [
        header(4, IT_INDIRECT_BUFFER, 0),
        0x400,
        0,
        4,
        header(2, IT_DRAW_INDEX_AUTO, 0),
        3,
    ];
    host.write(
        0x100,
        &outer
            .iter()
            .flat_map(|d| d.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let mut p = Processor::default();
    p.run(&host, 0x100, outer.len() as u32).unwrap();
    assert_eq!((p.sh[8], p.sh[9]), (0x1234, 0x5678));
    assert_eq!(p.stats.draws, 1);
}

#[test]
fn chained_indirect_buffers_continue_without_nesting() {
    let host = Fake::default();
    host.mem.replace(vec![0; 0x2000]);
    let segments = 2 * MAX_DEPTH as u64;
    for k in 0..segments {
        let at = 0x100 + 0x100 * k;
        let mut seg = vec![header(2, IT_DRAW_INDEX_AUTO, 0), 3];
        if k + 1 < segments {
            seg.extend([
                header(4, IT_INDIRECT_BUFFER, 0),
                (at + 0x100) as u32,
                0,
                7 | 1 << 20,
            ]);
            seg.extend([header(2, IT_DRAW_INDEX_AUTO, 0), 3]);
        }
        host.write(
            at,
            &seg.iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
    }
    let mut p = Processor::default();
    p.run(&host, 0x100, 8).unwrap();
    assert_eq!(p.stats.draws, segments);
}

#[test]
fn a_valid_rewind_runs_what_was_written_after_it() {
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    let first = [header(2, IT_REWIND, 0), REWIND_VALID];
    host.write(
        0x100,
        &first
            .iter()
            .flat_map(|d| d.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let late = [header(5, IT_WRITE_DATA, 0), 5 << 8, 0x800, 0, 9];
    struct Late<'a>(&'a Fake, [u32; 5]);
    impl Host for Late<'_> {
        fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
            if addr == 0x108 {
                self.0.write(
                    0x108,
                    &self
                        .1
                        .iter()
                        .flat_map(|d| d.to_le_bytes())
                        .collect::<Vec<u8>>(),
                );
            }
            self.0.read(addr, buf)
        }
        fn write(&self, addr: u64, data: &[u8]) -> bool {
            self.0.write(addr, data)
        }
        fn interrupt(&self, _: u32) {}
        fn flip(&self, _: Flip) {}
        fn exiting(&self) -> bool {
            false
        }
    }
    let mut p = Processor::default();
    p.run(&Late(&host, late), 0x100, 7).unwrap();
    assert_eq!(word(&host, 0x800), 9);
}

#[test]
fn cond_exec_skips_its_dwords_when_the_flag_is_zero() {
    let (host, _) = run(&[
        header(5, IT_COND_EXEC, 0),
        0x900,
        0,
        0,
        5,
        header(5, IT_WRITE_DATA, 0),
        5 << 8,
        0x800,
        0,
        1,
        header(5, IT_WRITE_DATA, 0),
        5 << 8,
        0x804,
        0,
        2,
    ]);
    assert_eq!((word(&host, 0x800), word(&host, 0x804)), (0, 2));
}

#[test]
fn copy_data_atomics_and_semaphores_update_memory() {
    let (host, _) = run(&[
        header(6, IT_COPY_DATA, 0),
        5 | 5 << 8,
        0x77,
        0,
        0x800,
        0,
        header(9, IT_ATOMIC_MEM, 0),
        15 | 0x40,
        0x800,
        0,
        3,
        0,
        0,
        0,
        0,
        header(4, IT_MEM_SEMAPHORE, 0),
        0x808,
        0,
        6 << 29,
        header(8, IT_RELEASE_MEM, 0),
        0,
        7 << 29,
        0x810,
        0,
        5,
        0,
        0,
    ]);
    assert_eq!(
        (word(&host, 0x800), word(&host, 0x808), word(&host, 0x810)),
        (0x7A, 1, 5)
    );
}

#[test]
fn a_semaphore_wait_takes_one_count() {
    let host = Fake::default();
    host.mem.replace(vec![0; 0x1000]);
    host.write(0x808, &2u64.to_le_bytes());
    let packets = [header(4, IT_MEM_SEMAPHORE, 0), 0x808, 0, 7 << 29];
    host.write(
        0x100,
        &packets
            .iter()
            .flat_map(|d| d.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let mut p = Processor::default();
    p.run(&host, 0x100, 4).unwrap();
    assert_eq!((word(&host, 0x808), p.stats.timeouts), (1, 0));
}

#[test]
fn waits_end_when_memory_matches_and_flips_are_passed_on() {
    let (host, p) = run(&[
        header(7, IT_WAIT_REG_MEM, 0),
        0x10 | 3,
        0x900,
        0,
        0,
        0xFFFF_FFFF,
        0,
        header(6, IT_NOP, NOP_FLIP),
        1,
        2,
        0,
        0x42,
        0,
    ]);
    assert_eq!(p.stats.timeouts, 0);
    assert_eq!(
        *host.flips.borrow(),
        [Flip {
            handle: 1,
            index: 2,
            mode: 0,
            arg: 0x42
        }]
    );
}

#[test]
fn saved_register_state_carries_over_to_a_new_processor() {
    let mut p = Processor::default();
    p.context[5] = 0x1234;
    p.sh[7] = 9;
    p.uconfig[0x100] = 0xabcd;
    p.index_base = 0x10_0000;
    p.instances = 3;
    p.gds[10] = 0x55;
    let saved = p.saved();
    let bytes = savestate::to_bytes(&saved);
    let back: Saved = savestate::from_bytes(&bytes).unwrap();
    assert_eq!(back, saved);
    let mut q = Processor::default();
    q.restore(back.clone(), false);
    assert_eq!(
        (q.context[5], q.sh[7], q.uconfig[0x100]),
        (0x1234, 9, 0xabcd)
    );
    assert_eq!((q.index_base, q.instances, q.gds[10]), (0x10_0000, 3, 0));
    q.restore(back, true);
    assert_eq!(q.gds[10], 0x55);
}
