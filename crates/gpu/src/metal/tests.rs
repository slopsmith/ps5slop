use super::*;
use std::cell::RefCell;

/// Guest memory for tests: one page-aligned host allocation at guest address 0.
struct Mem {
    bytes: RefCell<Vec<u8>>,
    base: *mut u8,
    len: u64,
    image: Option<(u64, u64)>,
    area: Option<(u64, u64)>,
    /// Mapped regions; empty means one region over the whole allocation.
    regions: Vec<(u64, u64)>,
}

impl Mem {
    fn new(len: u64) -> Self {
        // SAFETY: a fresh anonymous mapping.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        } as *mut u8;
        Self {
            bytes: RefCell::new(Vec::new()),
            base,
            len,
            image: None,
            area: None,
            regions: Vec::new(),
        }
    }
}

impl Host for Mem {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        let _ = &self.bytes;
        // SAFETY: tests stay inside the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(self.base.add(addr as usize), buf.as_mut_ptr(), buf.len())
        };
        true
    }
    fn write(&self, addr: u64, data: &[u8]) -> bool {
        // SAFETY: tests stay inside the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(addr as usize), data.len())
        };
        true
    }
    fn interrupt(&self, _: u32) {}
    fn flip(&self, _: crate::Flip) {}
    fn exiting(&self) -> bool {
        false
    }
    fn extent(&self, addr: u64) -> Option<(u64, u64)> {
        if self.regions.is_empty() {
            return (addr < self.len).then_some((0, self.len));
        }
        self.regions
            .iter()
            .copied()
            .find(|&(s, e)| s <= addr && addr < e)
    }
    fn image(&self, addr: u64) -> Option<(u64, u64)> {
        self.image.filter(|&(s, e)| s <= addr && addr < e)
    }
    fn gpu_area(&self) -> Option<(u64, u64)> {
        self.area
    }
    fn back(&self, _start: u64, end: u64) -> bool {
        end <= self.len
    }
    fn host_pointer(&self, addr: u64) -> Option<*mut u8> {
        // SAFETY: inside the mapping.
        Some(unsafe { self.base.add(addr as usize) })
    }
}

#[test]
fn written_ranges_follow_the_v_sharp_bounds() {
    // Base 0x10_0000_1000, stride 4, 0x100 records.
    let v = [0x1000, 0x10 | 4 << 16, 0x100, 0];
    assert_eq!(
        buffer_range(&v, true),
        (0x10_0000_1000, 0x10_0000_1000 + 0x404 + 16)
    );
    assert_eq!(
        buffer_range(&v, false),
        (0x10_0000_1000, 0x10_0000_1000 + 0x110)
    );
    let w = Written::Ranges(vec![buffer_range(&v, false)]);
    assert!(w.covers(0x10_0000_1100, 4));
    assert!(!w.covers(0x10_0000_1110, 4));
    assert!(!w.covers(0x10_0000_0ffc, 4));
    assert!(!Written::Nothing.covers(0, u64::MAX / 2));
    assert!(Written::Anything.covers(0, 1));
}

#[test]
fn the_htile_check_marks_cleared_words_and_sets_the_draw() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(1 << 20);
    mem.write(0x1000, &0x40u32.to_le_bytes());
    mem.write(0x1004, &0x45u32.to_le_bytes());
    let helpers = metal.htile_helpers().expect("the HTILE helpers compile");
    let (window, at) = metal.window(&mem, 0x1000, 8).unwrap();
    let (draws, d) = metal.scratch(32).unwrap();
    let cb = metal.command();
    let enc = cb.computeCommandEncoder().unwrap();
    enc.setComputePipelineState(&helpers.check);
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    for k in 0..2 {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&window), at + 4 * k, 0);
            enc.setBuffer_offset_atIndex(Some(&draws), d + 16 * k, 1);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, one);
    }
    enc.endEncoding();
    metal.flush_and_wait();
    let read = |a: u64| {
        let mut b = [0u8; 4];
        mem.read(a, &mut b);
        u32::from_le_bytes(b)
    };
    assert_eq!((read(0x1000), read(0x1004)), (0x4f, 0x45));
    let words = unsafe {
        std::slice::from_raw_parts(draws.contents().as_ptr().cast::<u32>().add(d / 4), 8)
    };
    assert_eq!(words, [3, 1, 0, 0, 0, 1, 0, 0]);
}

#[test]
fn integer_targets_take_integer_exports() {
    let mut c = vec![0u32; 0x400];
    for (n, number) in [(0, 4), (1, 7)] {
        c[context::CB_COLOR0_BASE + n * context::CB_COLOR_STRIDE] = 0x1000 * (n as u32 + 1);
        c[context::CB_COLOR0_INFO + n * context::CB_COLOR_STRIDE] = 12 << 2 | number << 8;
    }
    c[context::SPI_SHADER_COL_FORMAT] = 0x944;
    assert_eq!(export_formats(&c), 0x947);
}

/// Runs a compute program whose loop counts s2 up to the bound in s0 and
/// then appends to GDS; returns how many threads got there, how many
/// dispatches gave up, and the GPU time.
fn run_bounded_loop(
    metal: &mut Metal,
    mem: &Mem,
    code: &[u32],
    at: u64,
    groups: u32,
    bound: u32,
) -> (u32, u64, f64) {
    for (i, w) in code.iter().enumerate() {
        mem.write(at + 4 * i as u64, &w.to_le_bytes());
    }
    metal.with_gds(&mut |g| g.fill(0));
    let (c, u) = (vec![0u32; 0x400], vec![0u32; 0x400]);
    let mut sh = vec![0u32; 0x400];
    sh[sh::COMPUTE_PGM_LO] = (at >> 8) as u32;
    sh[sh::COMPUTE_PGM_RSRC2] = 1 << 1;
    sh[sh::COMPUTE_NUM_THREAD_X] = 64;
    sh[sh::COMPUTE_USER_DATA_0] = bound;
    let state = State {
        context: &c,
        sh: &sh,
        uconfig: &u,
        color_control_written: false,
    };
    let trips = metal.guard_trips;
    metal.dispatch(mem, &state, crate::Dispatch::groups([groups, 1, 1]));
    assert_eq!(metal.skipped, 0, "{:?}", metal.reasons);
    metal.commit();
    let start = std::time::Instant::now();
    let mut gpu = 0.0;
    for (cb, _, _) in std::mem::take(&mut metal.inflight) {
        while !matches!(
            cb.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        ) {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(5),
                "the loop did not end within 5 s"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(cb.error().is_none());
        gpu += cb.GPUEndTime() - cb.GPUStartTime();
    }
    metal.check_guards();
    let mut done = 0;
    metal.with_gds(&mut |g| done = u32::from_le_bytes(g[..4].try_into().unwrap()));
    (done, metal.guard_trips - trips, gpu)
}

// Each loop is also bounded by its user SGPR, so a guard that fails cannot
// hang the GPU: without the guard every case ends in well under a second.
#[test]
fn runaway_compute_loops_end_on_their_own() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(1 << 20);
    const HEAD: [u32; 3] = [
        0xBEFC_0380, // s_mov_b32 m0, 0
        0xBE82_0380, // s_mov_b32 s2, 0
        0x8002_8102, // s_add_u32 s2, s2, 1
    ];
    const TAIL: [u32; 5] = [
        0xBF0A_0002, // s_cmp_lt_u32 s2, s0
        0xBF85_FFFD, // s_cbranch_scc1 (to s_add_u32)
        0xD8FA_0000,
        0x0100_0000, // ds_append v1 gds
        0xBF81_0000, // s_endpgm
    ];
    let plain: Vec<u32> = HEAD.iter().chain(&TAIL).copied().collect();
    let mut barrier = HEAD.to_vec();
    barrier.push(0xBF8A_0000); // s_barrier
    barrier.extend(TAIL);
    barrier[5] = 0xBF85_FFFC;

    let (done, trips, gpu) = run_bounded_loop(&mut metal, &mem, &plain, 0x1000, 4, 1 << 12);
    assert_eq!(
        (done, trips),
        (256, 0),
        "the guard would cut off loops that end on their own"
    );
    eprintln!("honest: {gpu:.4} s");
    let (done, trips, gpu) = run_bounded_loop(&mut metal, &mem, &plain, 0x1000, 1, 1 << 25);
    assert_eq!(
        (done, trips),
        (0, 1),
        "a runaway in one group would hang the GPU"
    );
    assert!(
        gpu < 0.1,
        "a runaway would stall other Metal work for {gpu} s"
    );
    eprintln!("one group: {gpu:.4} s");
    let (done, trips, gpu) = run_bounded_loop(&mut metal, &mem, &plain, 0x1000, 1 << 14, 1 << 13);
    assert_eq!(trips, 1);
    assert!(
        done < (1 << 20) / 2,
        "a runaway grid would keep the GPU busy: {done} of {} threads finished",
        1 << 20
    );
    assert!(
        gpu < 1.0,
        "a runaway would stall other Metal work for {gpu} s"
    );
    eprintln!("large grid: {gpu:.4} s, {done} finished");
    let (done, trips, gpu) = run_bounded_loop(&mut metal, &mem, &barrier, 0x2000, 2, 1 << 25);
    assert_eq!(
        (done, trips),
        (0, 1),
        "a group whose threads leave one by one would wait at its barrier forever"
    );
    assert!(
        gpu < 0.1,
        "a barrier runaway would stall other Metal work for {gpu} s"
    );
    eprintln!("barrier: {gpu:.4} s");
}

fn word(mem: &Mem, at: u64) -> u32 {
    let mut b = [0; 4];
    mem.read(at, &mut b);
    u32::from_le_bytes(b)
}

/// Registers that dispatch one thread of the program at `code` with `user` data.
fn compute_setup(code: u64, user: &[u32]) -> Vec<u32> {
    let set = |reg: usize, values: &[u32]| {
        let mut p = vec![
            0xC000_0000 | (values.len() as u32) << 16 | 0x76 << 8,
            reg as u32,
        ];
        p.extend_from_slice(values);
        p
    };
    [
        set(sh::COMPUTE_PGM_LO, &[(code >> 8) as u32, 0]),
        set(sh::COMPUTE_PGM_RSRC2, &[(user.len() as u32) << 1]),
        set(sh::COMPUTE_NUM_THREAD_X, &[1, 1, 1]),
        set(sh::COMPUTE_USER_DATA_0, user),
    ]
    .concat()
}

fn write_data(dest: u64, value: u32) -> [u32; 5] {
    [0xC003_3700, 5 << 8, dest as u32, (dest >> 32) as u32, value]
}

const DISPATCH_ONE: [u32; 5] = [0xC003_1500, 1, 1, 1, 1];

pub(super) fn buffer(base: u64) -> [u32; 4] {
    [base as u32, (base >> 32) as u32, 4, 0x5204]
}

#[test]
fn writes_from_the_command_stream_land_between_the_dispatches_around_them() {
    let Some(metal) = Metal::new() else {
        return;
    };
    let mut mem = Mem::new(1 << 20);
    mem.area = Some((0, 1 << 20));
    let (a, b) = (0x8000, 0x8100);
    let copy: [u32; 6] = [
        0xE030_0000,
        0x8000_0100, // buffer_load_dword v1, off, s[0:3], 0
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_0000,
        0x8001_0100, // buffer_store_dword v1, off, s[4:7], 0
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in copy.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let packets = [
        compute_setup(0x1000, &[buffer(a), buffer(b)].concat()),
        write_data(a, 7).to_vec(),
        DISPATCH_ONE.to_vec(),
        write_data(a, 9).to_vec(),
    ]
    .concat();
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    mem.write(0x2000, &bytes);
    let mut p = crate::pm4::Processor::default();
    p.backend = Some(Box::new(metal));
    p.run(&mem, 0x2000, packets.len() as u32).unwrap();
    p.finish(&mem);
    assert_eq!(
        (word(&mem, b), word(&mem, a)),
        (7, 9),
        "the dispatch must read the write before it, and the write after it must not run first"
    );
}

#[test]
fn an_indirect_dispatch_in_threads_runs_the_threads_earlier_gpu_work_asked_for() {
    let Some(metal) = Metal::new() else {
        return;
    };
    let mut mem = Mem::new(1 << 20);
    mem.area = Some((0, 1 << 20));
    let (dims, out) = (0x8000u64, 0x9000u64);
    mem.write(dims, &[0u32, 1, 1].map(u32::to_le_bytes).concat());
    let size: &[u32] = &[
        0x7E02_02FF,
        0x0000_0046, // v_mov_b32 v1, 70
        0xE070_0000,
        0x8000_0100, // buffer_store_dword v1, off, s[0:3], 0
        0xBF81_0000, // s_endpgm
    ];
    let mark: &[u32] = &[
        0xD746_0001,
        0x0401_0C04, // v_lshl_add_u32 v1, s4, 6, v0
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x7E04_0281, // v_mov_b32 v2, 1
        0xE070_1000,
        0x8000_0201, // buffer_store_dword v2, v1, s[0:3], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    for (at, code) in [(0x1000u64, size), (0x1100, mark)] {
        for (i, w) in code.iter().enumerate() {
            mem.write(at + 4 * i as u64, &w.to_le_bytes());
        }
    }
    let set = |reg: usize, values: &[u32]| {
        let mut p = vec![
            0xC000_0000 | (values.len() as u32) << 16 | 0x76 << 8,
            reg as u32,
        ];
        p.extend_from_slice(values);
        p
    };
    let packets = [
        compute_setup(0x1000, &buffer(dims)),
        DISPATCH_ONE.to_vec(),
        set(sh::COMPUTE_PGM_LO, &[0x11, 0]),
        set(sh::COMPUTE_PGM_RSRC2, &[4 << 1 | 1 << 7]),
        set(sh::COMPUTE_NUM_THREAD_X, &[64, 1, 1]),
        set(sh::COMPUTE_USER_DATA_0, &[out as u32, 0, 0x1000, 0x5204]),
        vec![0xC002_1600, dims as u32, 0, 0x21],
    ]
    .concat();
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    mem.write(0x2000, &bytes);
    let mut p = crate::pm4::Processor::default();
    p.backend = Some(Box::new(metal));
    p.run(&mem, 0x2000, packets.len() as u32).unwrap();
    p.finish(&mem);
    assert_eq!(p.stats.gpu_indirect, 1);
    let marked: Vec<u32> = (0..128).map(|i| word(&mem, out + 4 * i)).collect();
    assert_eq!(marked, [[1; 70].as_slice(), &[0; 58]].concat());
}

#[test]
fn an_indirect_dispatch_past_the_group_limit_runs_nothing() {
    let Some(metal) = Metal::new() else {
        return;
    };
    let mut mem = Mem::new(1 << 20);
    mem.area = Some((0, 1 << 20));
    let (dims, out) = (0x8000u64, 0x9000u64);
    mem.write(dims, &[1u32 << 25, 1, 1].map(u32::to_le_bytes).concat());
    let mark: &[u32] = &[
        0x7E04_0281, // v_mov_b32 v2, 1
        0xE070_0000,
        0x8000_0200, // buffer_store_dword v2, off, s[0:3], 0
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in mark.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let packets = [
        compute_setup(0x1000, &buffer(out)),
        vec![0xC002_1600, dims as u32, 0, 1],
    ]
    .concat();
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    mem.write(0x2000, &bytes);
    let mut p = crate::pm4::Processor::default();
    p.backend = Some(Box::new(metal));
    p.run(&mem, 0x2000, packets.len() as u32).unwrap();
    p.finish(&mem);
    assert_eq!(p.stats.gpu_indirect, 1);
    assert_eq!(word(&mem, out), 0);
}

#[test]
fn float_min_and_64_bit_atomics_change_memory_and_lds_as_on_hardware() {
    let Some(metal) = Metal::new() else {
        return;
    };
    let mut mem = Mem::new(1 << 20);
    mem.area = Some((0, 1 << 20));
    let (a, b) = (0x8000u64, 0x8100u64);
    mem.write(a, &2.0f32.to_le_bytes());
    mem.write(b, &[0xF0u32.to_le_bytes(), 0xF00u32.to_le_bytes()].concat());
    let (c, d) = (0x8200u64, 0x8300u64);
    let program: &[u32] = &[
        0x7E08_0280, // v_mov_b32 v4, 0
        0x7E02_02FF,
        0x4000_0000, // v_mov_b32 v1, 2.0
        0xD834_0000,
        0x0000_0104, // ds_write_b32 v4, v1
        0x7E02_02FF,
        0x3FC0_0000, // v_mov_b32 v1, 1.5
        0xD848_0000,
        0x0000_0104, // ds_min_f32 v4, v1
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xD8D8_0000,
        0x0500_0004, // ds_read_b32 v5, v4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_0000,
        0x8002_0500, // buffer_store_dword v5, off, s[8:11], 0
        0x7E02_02FF,
        0x3FC0_0000, // v_mov_b32 v1, 1.5
        0xE0FC_0000,
        0x8000_0100, // buffer_atomic_fmin v1, off, s[0:3], 0
        0x7E02_02FF,
        0x4040_0000, // v_mov_b32 v1, 3.0
        0xE0FC_0000,
        0x8000_0100, // buffer_atomic_fmin v1, off, s[0:3], 0
        0x7E04_02FF,
        0x0000_000F, // v_mov_b32 v2, 0xf
        0x7E06_02FF,
        0x0000_00F0, // v_mov_b32 v3, 0xf0
        0xE168_0000,
        0x8001_0200, // buffer_atomic_or_x2 v[2:3], off, s[4:7], 0
        0xE140_0000,
        0x8003_0200, // buffer_atomic_swap_x2 v[2:3], off, s[12:15], 0
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in program.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let packets = [
        compute_setup(
            0x1000,
            &[buffer(a), buffer(b), buffer(c), buffer(d)].concat(),
        ),
        DISPATCH_ONE.to_vec(),
    ]
    .concat();
    let mut packets = packets;
    // COMPUTE_PGM_RSRC2's value: LDS_SIZE of one 512-byte block.
    packets[6] |= 1 << 15;
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    mem.write(0x2000, &bytes);
    let mut p = crate::pm4::Processor::default();
    p.backend = Some(Box::new(metal));
    p.run(&mem, 0x2000, packets.len() as u32).unwrap();
    p.finish(&mem);
    assert_eq!(f32::from_bits(word(&mem, a)), 1.5);
    assert_eq!((word(&mem, b), word(&mem, b + 4)), (0xFF, 0xFF0));
    assert_eq!(f32::from_bits(word(&mem, c)), 1.5, "ds_min_f32");
    assert_eq!((word(&mem, d), word(&mem, d + 4)), (0xF, 0xF0), "swap_x2");
}

// A lane-0 ticket broadcast with `v_readlane`, as GTA V's look-back scans take one.
#[test]
fn one_wave64_group_sees_whole_wave_masks_and_lanes() {
    let Some(metal) = Metal::new() else {
        return;
    };
    let mut mem = Mem::new(1 << 20);
    mem.area = Some((0, 1 << 20));
    let (a, b) = (0x8000u64, 0x9000u64);
    mem.write(a, &5u32.to_le_bytes());
    let program: &[u32] = &[
        0x8790_817E, // s_and_b64 s[16:17], exec, 1
        0xBEFE_0410, // s_mov_b64 exec, s[16:17]
        0x7E02_0281, // v_mov_b32_e32 v1, 1
        0xE0C8_4000,
        0x8000_0100, // buffer_atomic_add v1, off, s[0:3], 0 glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xD760_0008,
        0x0001_0101, // v_readlane_b32 s8, v1, 0
        0xBEFE_04C1, // s_mov_b64 exec, -1
        0xD765_0002,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v2, -1, 0
        0xD766_0002,
        0x0002_04C1, // v_mbcnt_hi_u32_b32 v2, -1, v2
        0x7D88_04A8, // v_cmp_gt_u32_e32 vcc, 40, v2
        0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
        0xD760_000A,
        0x0001_7F02, // v_readlane_b32 s10, v2, 63
        0x8F08_9808, // s_lshl_b32 s8, s8, 24
        0x8F09_9009, // s_lshl_b32 s9, s9, 16
        0x8F0A_880A, // s_lshl_b32 s10, s10, 8
        0x8808_0908, // s_or_b32 s8, s8, s9
        0x8808_0A08, // s_or_b32 s8, s8, s10
        0x3806_0408, // v_or_b32_e32 v3, s8, v2
        0x3408_0482, // v_lshlrev_b32_e32 v4, 2, v2
        0xE070_1000,
        0x8001_0304, // buffer_store_dword v3, v4, s[4:7], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in program.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let out = [b as u32, (b >> 32) as u32, 256, 0x5204];
    let mut packets = [
        compute_setup(0x1000, &[buffer(a), out].concat()),
        DISPATCH_ONE.to_vec(),
    ]
    .concat();
    // COMPUTE_NUM_THREAD_X: one wave64 group.
    packets[9] = 64;
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    mem.write(0x2000, &bytes);
    let mut p = crate::pm4::Processor::default();
    p.backend = Some(Box::new(metal));
    p.run(&mem, 0x2000, packets.len() as u32).unwrap();
    p.finish(&mem);
    assert_eq!(word(&mem, a), 6, "only lane 0 takes a ticket");
    let lanes: Vec<u32> = (0..64).map(|i| word(&mem, b + 4 * i)).collect();
    let want: Vec<u32> = (0..64).map(|i| 5 << 24 | 40 << 16 | 63 << 8 | i).collect();
    assert_eq!(lanes, want);
}

#[test]
fn programs_do_not_store_into_a_loaded_module() {
    let store: [u32; 4] = [
        0x7E02_0204, // v_mov_b32 v1, s4
        0xE070_0000,
        0x8000_0100, // buffer_store_dword v1, off, s[0:3], 0
        0xBF81_0000, // s_endpgm
    ];
    let mut results = Vec::new();
    for image in [None, Some((0, 1 << 20))] {
        let Some(metal) = Metal::new() else {
            return;
        };
        let mut mem = Mem::new(1 << 20);
        mem.image = image;
        for (i, w) in store.iter().enumerate() {
            mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
        }
        let mut p = crate::pm4::Processor::default();
        let packets = [
            compute_setup(0x1000, &[&buffer(0x8000)[..], &[0x1234]].concat()),
            DISPATCH_ONE.to_vec(),
        ]
        .concat();
        let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
        mem.write(0x2000, &bytes);
        p.backend = Some(Box::new(metal));
        p.run(&mem, 0x2000, packets.len() as u32).unwrap();
        p.finish(&mem);
        results.push(word(&mem, 0x8000));
    }
    assert_eq!(results, [0x1234, 0]);
}

#[test]
fn a_program_reads_a_table_in_a_third_region() {
    let Some(metal) = Metal::new() else {
        return;
    };
    let (code, table, out) = (0x1000, 0x400_0000, 0x800_0000);
    let mut mem = Mem::new(0x1000_0000);
    mem.regions = vec![(0, 0x100_0000), (table, 0x600_0000), (out, 0x1000_0000)];
    let program: [u32; 7] = [
        0xF400_0200,
        0xFA00_0000, // s_load_dword s8, s[0:1], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_0208, // v_mov_b32 v1, s8
        0xE070_0000,
        0x8001_0100, // buffer_store_dword v1, off, s[4:7], 0
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in program.iter().enumerate() {
        mem.write(code + 4 * i as u64, &w.to_le_bytes());
    }
    mem.write(table, &0x5EED_u32.to_le_bytes());
    let mut p = crate::pm4::Processor::default();
    let user = [&[table as u32, 0, 0, 0][..], &buffer(out)[..]].concat();
    let packets = [compute_setup(code, &user), DISPATCH_ONE.to_vec()].concat();
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    mem.write(0x2000, &bytes);
    p.backend = Some(Box::new(metal));
    p.run(&mem, 0x2000, packets.len() as u32).unwrap();
    p.finish(&mem);
    assert_eq!(word(&mem, out), 0x5EED);
}

// The poll is a guarded loop, so a broken translation ends with a guard
// trip instead of hanging the GPU.
#[test]
fn a_group_polling_a_glc_flag_sees_another_group_set_it() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(1 << 20);
    let code: [u32; 16] = [
        0xBF06_8004, // s_cmp_eq_u32 s4, 0
        0xBF85_000A, // s_cbranch_scc1 (to the store)
        0xE030_4000,
        0x8000_0100, // buffer_load_dword v1, off, s[0:3], 0 glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E0A_0501, // v_readfirstlane_b32 s5, v1
        0xBF06_8005, // s_cmp_eq_u32 s5, 0
        0xBF85_FFFA, // s_cbranch_scc1 (to the load)
        0x7E04_0282, // v_mov_b32 v2, 2
        0xE070_0004,
        0x8000_0200, // buffer_store_dword v2, off, s[0:3], 0 offset:4
        0xBF81_0000, // s_endpgm
        0x7E04_0281, // v_mov_b32 v2, 1
        0xE070_4000,
        0x8000_0200, // buffer_store_dword v2, off, s[0:3], 0 glc
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in code.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let (c, u) = (vec![0u32; 0x400], vec![0u32; 0x400]);
    let mut sh = vec![0u32; 0x400];
    sh[sh::COMPUTE_PGM_LO] = 0x10;
    sh[sh::COMPUTE_PGM_RSRC2] = 4 << 1 | 1 << 7;
    sh[sh::COMPUTE_NUM_THREAD_X] = 1;
    sh[sh::COMPUTE_USER_DATA_0..sh::COMPUTE_USER_DATA_0 + 4]
        .copy_from_slice(&[0x8000, 0, 8, 0x5204]);
    let state = State {
        context: &c,
        sh: &sh,
        uconfig: &u,
        color_control_written: false,
    };
    metal.dispatch(&mem, &state, crate::Dispatch::groups([2, 1, 1]));
    assert_eq!(metal.skipped, 0, "{:?}", metal.reasons);
    metal.flush_and_wait();
    metal.check_guards();
    assert_eq!(
        (word(&mem, 0x8000), word(&mem, 0x8004), metal.guard_trips),
        (1, 2, 0),
        "the polling group never saw the flag"
    );
}

#[test]
fn swapped_10_10_10_2_targets_keep_the_hardware_component_order() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(4 << 20);
    let target = ColorTarget {
        address: 0x10000,
        width: 64,
        height: 32,
        format: regs::ColorFormat {
            layout: 9,
            number: 0,
            swap: 1,
        },
        tile: Some(TileMode::RenderTarget),
        raw_tile: 0x1b,
        max_mip: 0,
        mip_level: 0,
        slice: 0,
        slices: 1,
        fast_clear: None,
        dcc: None,
    };
    let (drawn, format) = metal.target(&mem, &target).expect("the target is made");
    assert_eq!(format, MTLPixelFormat::BGR10A2Unorm);
    let image = Image {
        address: target.address,
        format: 50,
        width: 64,
        height: 32,
        swizzle: [6, 5, 4, 7],
        tile: Some(TileMode::RenderTarget),
        raw_tile: 0x1b,
        kind: 9,
        base_level: 0,
        last_level: 0,
        depth: 1,
        base_array: 0,
    };
    let read = metal.image(&mem, image, TextureKind::Tex2d).unwrap();
    assert_eq!(read.pixelFormat(), MTLPixelFormat::RGB10A2Unorm);
    let parent = read.parentTexture().map(|p| Retained::as_ptr(&p));
    assert_eq!(parent, Some(Retained::as_ptr(&drawn)));
}

#[test]
fn sampling_reads_what_was_written_last_at_an_address() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(4 << 20);
    let (w, h) = (64u32, 32u32);
    let array = Image {
        address: 0x10000,
        format: 56,
        width: w,
        height: h,
        swizzle: [4, 5, 6, 7],
        tile: Some(TileMode::RenderTarget),
        raw_tile: 0x1b,
        kind: 13,
        base_level: 0,
        last_level: 0,
        depth: 2,
        base_array: 0,
    };
    let flat = Image {
        kind: 9,
        depth: 1,
        ..array
    };
    let target = ColorTarget {
        address: array.address,
        width: w,
        height: h,
        format: regs::ColorFormat {
            layout: 10,
            number: 0,
            swap: 0,
        },
        tile: Some(TileMode::RenderTarget),
        raw_tile: 0x1b,
        max_mip: 0,
        mip_level: 0,
        slice: 0,
        slices: 1,
        fast_clear: None,
        dcc: None,
    };
    let volume = metal
        .storage_image(&mem, array, TextureKind::Tex2dArray)
        .expect("the 2D array is made");
    let parent = |t: &Texture| t.parentTexture().map(|p| Retained::as_ptr(&p));
    let volume = parent(&volume).expect("a view of the 2D array");
    let (drawn, _) = metal.target(&mem, &target).expect("the target is made");
    let read = metal.image(&mem, flat, TextureKind::Tex2d).unwrap();
    assert_eq!(Retained::as_ptr(&read), Retained::as_ptr(&drawn));
    let rewritten = metal
        .storage_image(&mem, array, TextureKind::Tex2dArray)
        .unwrap();
    assert_eq!(parent(&rewritten), Some(volume));
    let read = metal.image(&mem, flat, TextureKind::Tex2d).unwrap();
    assert_eq!(parent(&read), Some(volume));
}

fn first_texel(metal: &mut Metal, tex: &Texture) -> [u8; 4] {
    let d = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            tex.pixelFormat(),
            tex.width(),
            tex.height(),
            false,
        )
    };
    d.setStorageMode(MTLStorageMode::Shared);
    let shared = metal.device.newTextureWithDescriptor(&d).unwrap();
    let cb = metal.command();
    let blit = cb.blitCommandEncoder().unwrap();
    unsafe { blit.copyFromTexture_toTexture(tex, &shared) };
    blit.endEncoding();
    metal.flush_and_wait();
    let mut out = [0u8; 4];
    unsafe {
        shared.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            NonNull::new_unchecked(out.as_mut_ptr().cast()),
            4 * tex.width(),
            MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            },
            0,
        );
    }
    out
}

#[test]
fn an_image_follows_guest_memory_from_frame_to_frame() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(4 << 20);
    let image = Image {
        address: 0x10000,
        format: 56,
        width: 64,
        height: 32,
        swizzle: [4, 5, 6, 7],
        tile: Some(TileMode::RenderTarget),
        raw_tile: 0x1b,
        kind: 9,
        base_level: 0,
        last_level: 0,
        depth: 1,
        base_array: 0,
    };
    let mut seen = Vec::new();
    for texel in [[1, 2, 3, 4], [1, 2, 3, 4], [9, 8, 7, 6], [9, 8, 7, 6]] {
        mem.write(image.address, &texel);
        let tex = metal.image(&mem, image, TextureKind::Tex2d).unwrap();
        seen.push(first_texel(&mut metal, &tex));
        metal.frame_done(None);
    }
    assert_eq!(
        seen,
        [[1, 2, 3, 4], [1, 2, 3, 4], [9, 8, 7, 6], [9, 8, 7, 6]]
    );
}

#[test]
fn a_buffer_store_over_a_whole_image_makes_its_next_use_read_memory() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(4 << 20);
    let store: [u32; 4] = [
        0x7E02_0204, // v_mov_b32 v1, s4
        0xE070_0000,
        0x8000_0100, // buffer_store_dword v1, off, s[0:3], 0
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in store.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let image = Image {
        address: 0x10000,
        format: 56,
        width: 64,
        height: 32,
        swizzle: [4, 5, 6, 7],
        tile: Some(TileMode::RenderTarget),
        raw_tile: 0x1b,
        kind: 9,
        base_level: 0,
        last_level: 0,
        depth: 1,
        base_array: 0,
    };
    let bytes = 64 * 32 * image_format(56).unwrap().1;
    let mut kept = Vec::new();
    for records in [bytes, 16 * bytes] {
        let made = metal
            .storage_image(&mem, image, TextureKind::Tex2d)
            .unwrap();
        let (c, u) = (vec![0u32; 0x400], vec![0u32; 0x400]);
        let mut sh = vec![0u32; 0x400];
        sh[sh::COMPUTE_PGM_LO] = 0x10;
        sh[sh::COMPUTE_PGM_RSRC2] = 5 << 1;
        sh[sh::COMPUTE_NUM_THREAD_X] = 1;
        sh[sh::COMPUTE_USER_DATA_0..sh::COMPUTE_USER_DATA_0 + 5]
            .copy_from_slice(&[0x10000, 0, records, 0x5204, 7]);
        let state = State {
            context: &c,
            sh: &sh,
            uconfig: &u,
            color_control_written: false,
        };
        metal.dispatch(&mem, &state, crate::Dispatch::groups([1, 1, 1]));
        let next = metal
            .storage_image(&mem, image, TextureKind::Tex2d)
            .unwrap();
        kept.push(Retained::as_ptr(&made) == Retained::as_ptr(&next));
    }
    metal.flush_and_wait();
    assert_eq!(kept, [false, true], "an arena-sized store keeps the image");
}

#[test]
fn depth_clears_reach_past_a_smaller_color_target() {
    let mut cleared = Vec::new();
    for on_gpu in [false, true] {
        let Some(mut metal) = Metal::new() else {
            return;
        };
        let mem = Mem::new(1 << 20);
        let texture = |format, w, h| {
            let d = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    format, w, h, false,
                )
            };
            d.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            d.setStorageMode(MTLStorageMode::Private);
            metal.device.newTextureWithDescriptor(&d).unwrap()
        };
        let color = texture(MTLPixelFormat::RGBA8Unorm, 32, 32);
        let depth = texture(MTLPixelFormat::Depth32Float, 64, 32);
        let targets = [(0x10000, color, 0)];
        let d = (0x20000, depth.clone(), false, 0);
        if on_gpu {
            metal
                .htile_on_gpu(&mem, 0x30000, &targets, &d, None, (0.25, 0))
                .unwrap();
        } else {
            metal.pass(&targets, Some(d), Some((0.25, 0))).unwrap();
        }
        metal.end_pass();
        let out = metal
            .device
            .newBufferWithLength_options(64 * 32 * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let cb = metal.command();
        let blit = cb.blitCommandEncoder().unwrap();
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    &depth, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: 64, height: 32, depth: 1 }, &out, 0, 64 * 4, 64 * 32 * 4,
                );
        }
        blit.endEncoding();
        metal.flush_and_wait();
        let z =
            |x: usize, y: usize| unsafe { *out.contents().as_ptr().cast::<f32>().add(y * 64 + x) };
        cleared.push((z(8, 8), z(56, 8)));
    }
    assert_eq!(cleared, [(0.25, 0.25), (0.25, 0.25)]);
}

#[test]
fn nested_loops_with_early_exits_run_as_on_hardware() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(1 << 20);
    let code: [u32; 18] = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0383, // s_mov_b32 s6, 3
        0xBE87_0384, // outer: s_mov_b32 s7, 4
        0x8005_8105, // inner: s_add_u32 s5, s5, 1
        0xBF06_8206, // s_cmp_eq_u32 s6, 2
        0xBF84_0002, // s_cbranch_scc0 (to cont)
        0xBF06_8207, // s_cmp_eq_u32 s7, 2
        0xBF85_0003, // s_cbranch_scc1 (out of inner)
        0x8087_8107, // cont: s_sub_u32 s7, s7, 1
        0xBF07_8007, // s_cmp_lg_u32 s7, 0
        0xBF85_FFF8, // s_cbranch_scc1 (to inner)
        0x8086_8106, // s_sub_u32 s6, s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_FFF4, // s_cbranch_scc1 (to outer)
        0x7E04_0205, // v_mov_b32 v2, s5
        0xE070_0000,
        0x8000_0200, // buffer_store_dword v2, off, s[0:3], 0
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in code.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let (c, u) = (vec![0u32; 0x400], vec![0u32; 0x400]);
    let mut sh = vec![0u32; 0x400];
    sh[sh::COMPUTE_PGM_LO] = 0x10;
    sh[sh::COMPUTE_PGM_RSRC2] = 4 << 1;
    sh[sh::COMPUTE_NUM_THREAD_X] = 1;
    sh[sh::COMPUTE_USER_DATA_0..sh::COMPUTE_USER_DATA_0 + 4]
        .copy_from_slice(&[0x8000, 0, 8, 0x5204]);
    let state = State {
        context: &c,
        sh: &sh,
        uconfig: &u,
        color_control_written: false,
    };
    metal.dispatch(&mem, &state, crate::Dispatch::groups([1, 1, 1]));
    assert_eq!(metal.skipped, 0, "{:?}", metal.reasons);
    metal.flush_and_wait();
    assert_eq!(word(&mem, 0x8000), 11);
}

#[test]
fn packing_halves_rounds_toward_zero() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(1 << 20);
    let code: [u32; 11] = [
        0xBE85_03FF,
        0x4974_2400, // s_mov_b32 s5, 1e6
        0xBE86_03FF,
        0x3F80_1FFF, // s_mov_b32 s6, just under 1 + 2^-10
        0xD52F_0002,
        0x0000_0C05, // v_cvt_pkrtz_f16_f32 v2, s5, s6
        0xE070_0000,
        0x8000_0200, // buffer_store_dword v2, off, s[0:3], 0
        0xBF81_0000, // s_endpgm
        0,
        0,
    ];
    for (i, w) in code.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let (c, u) = (vec![0u32; 0x400], vec![0u32; 0x400]);
    let mut sh = vec![0u32; 0x400];
    sh[sh::COMPUTE_PGM_LO] = 0x10;
    sh[sh::COMPUTE_PGM_RSRC2] = 4 << 1;
    sh[sh::COMPUTE_NUM_THREAD_X] = 1;
    sh[sh::COMPUTE_USER_DATA_0..sh::COMPUTE_USER_DATA_0 + 4]
        .copy_from_slice(&[0x8000, 0, 8, 0x5204]);
    let state = State {
        context: &c,
        sh: &sh,
        uconfig: &u,
        color_control_written: false,
    };
    metal.dispatch(&mem, &state, crate::Dispatch::groups([1, 1, 1]));
    assert_eq!(metal.skipped, 0, "{:?}", metal.reasons);
    metal.flush_and_wait();
    assert_eq!(word(&mem, 0x8000), 0x3C00_7BFF);
}

#[test]
fn half_operations_select_and_keep_the_halves_they_should() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(1 << 20);
    let code: [u32; 26] = [
        0xBE85_03FF,
        0x4000_3C00, // s_mov_b32 s5, halves 1.0 and 2.0
        0xBE86_03FF,
        0x4400_4200, // s_mov_b32 s6, halves 3.0 and 4.0
        0x7E04_0205, // v_mov_b32 v2, s5
        0x7E06_0206, // v_mov_b32 v3, s6
        0x7E0A_0206, // v_mov_b32 v5, s6
        0x7E0C_0205, // v_mov_b32 v6, s5
        0x7E0E_0280, // v_mov_b32 v7, 0
        0xCC0E_4004,
        0x1C0A_0702, // v_pk_fma_f16 v4, v2, v3, v2
        0xD74B_5805,
        0x040A_0702, // v_fma_f16 v5, v2, v3, v2 op_sel:[1,1,0,1]
        0xD703_0806,
        0x0001_8302, // v_add_nc_u16 v6, v2, -1 op_sel:[1,0,0]
        0x6A0E_06F9,
        0x0405_1502, // v_mul_f16_sdwa v7, v2, v3 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_0
        0xD757_0007,
        0x0409_E103, // v_med3_f16 v7, v3, 0.5, v2
        0xE078_0000,
        0x8000_0400, // buffer_store_dwordx4 v[4:7], off, s[0:3], 0
        0xBF81_0000, // s_endpgm
        0,
        0,
        0,
        0,
    ];
    for (i, w) in code.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let (c, u) = (vec![0u32; 0x400], vec![0u32; 0x400]);
    let mut sh = vec![0u32; 0x400];
    sh[sh::COMPUTE_PGM_LO] = 0x10;
    sh[sh::COMPUTE_PGM_RSRC2] = 4 << 1;
    sh[sh::COMPUTE_NUM_THREAD_X] = 1;
    sh[sh::COMPUTE_USER_DATA_0..sh::COMPUTE_USER_DATA_0 + 4]
        .copy_from_slice(&[0x8000, 0, 16, 0x5204]);
    let state = State {
        context: &c,
        sh: &sh,
        uconfig: &u,
        color_control_written: false,
    };
    metal.dispatch(&mem, &state, crate::Dispatch::groups([1, 1, 1]));
    assert_eq!(metal.skipped, 0, "{:?}", metal.reasons);
    metal.flush_and_wait();
    let got: Vec<u32> = (0..4).map(|i| word(&mem, 0x8000 + 4 * i)).collect();
    assert_eq!(got, [0x4900_4400, 0x4880_4200, 0x4000_3FFF, 0x4600_3C00]);
}

#[test]
fn detiling_a_render_target_on_the_gpu_matches_the_cpu_layout() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(4 << 20);
    let (w, h) = (300u32, 200u32);
    let layout = Layout::new(TileMode::RenderTarget, 4, w, h).unwrap();
    for y in 0..h {
        for x in 0..w {
            mem.write(layout.offset(x, y), &(x | y << 16).to_le_bytes());
        }
    }
    let tex = metal
        .new_texture(MTLPixelFormat::R32Uint, w, h, false)
        .unwrap();
    assert!(metal.detile_into(&mem, &tex, 0, 0, layout));
    let shared = {
        let d = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R32Uint,
                w as usize,
                h as usize,
                false,
            )
        };
        d.setStorageMode(MTLStorageMode::Shared);
        metal.device.newTextureWithDescriptor(&d).unwrap()
    };
    let cb = metal.command();
    let blit = cb.blitCommandEncoder().unwrap();
    unsafe { blit.copyFromTexture_toTexture(&tex, &shared) };
    blit.endEncoding();
    metal.flush_and_wait();
    let mut out = vec![0u32; (w * h) as usize];
    unsafe {
        shared.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            NonNull::new_unchecked(out.as_mut_ptr().cast()),
            w as usize * 4,
            MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: w as usize,
                    height: h as usize,
                    depth: 1,
                },
            },
            0,
        );
    }
    for y in 0..h {
        for x in 0..w {
            assert_eq!(out[(y * w + x) as usize], x | y << 16);
        }
    }
}

#[test]
fn indexed_triangle_fan_renders_and_pixel_addtid_spills_round_trip() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    metal.set_wait_for_compiles(true);
    let mem = Mem::new(1 << 20);
    let (vs_addr, ps_addr, vb_addr, ib_addr, target_addr) = (
        0x1_0000u64,
        0x1_4000u64,
        0x2_0000u64,
        0x2_4000u64,
        0x4_0000u64,
    );
    let (w, h) = (16u32, 16u32);

    // Legacy VS: loads float4 clip pos from V# in s[0:3] using vertex_id in v0.
    let vs: [u32; 6] = [
        0xE038_2000,
        0x8000_0100, // buffer_load_dwordx4 v[1:4], v0, s[0:3], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_08CF,
        0x0403_0201, // exp pos0 v1, v2, v3, v4 done
        0xBF81_0000, // s_endpgm
    ];
    // PS: spills s0 (R) and s1 (G) through ds_write_addtid_b32 / ds_read_addtid_b32, exports RGBA.
    let ps: [u32; 14] = [
        0xBEFC_0380, // s_mov_b32 m0, 0
        0x7E00_0200, // v_mov_b32 v0, s0
        0x7E02_0201, // v_mov_b32 v1, s1
        0xDAC0_0000,
        0x0000_0000, // ds_write_addtid_b32 v0
        0xDAC0_0100,
        0x0000_0100, // ds_write_addtid_b32 v1 offset:256
        0xDAC4_0000,
        0x0200_0000, // ds_read_addtid_b32 v2
        0xDAC4_0100,
        0x0300_0000, // ds_read_addtid_b32 v3 offset:256
        0xF800_180F,
        0x0302_0302, // exp mrt0 v2, v3, v2, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in vs.iter().enumerate() {
        mem.write(vs_addr + 4 * i as u64, &w.to_le_bytes());
    }
    for (i, w) in ps.iter().enumerate() {
        mem.write(ps_addr + 4 * i as u64, &w.to_le_bytes());
    }
    // 4 quad corners covering [-1, 1]^2, indexed as a 16-bit triangle fan (0, 1, 2, 3).
    let verts: [[f32; 4]; 4] = [
        [-1.0, -1.0, 0.5, 1.0],
        [1.0, -1.0, 0.5, 1.0],
        [1.0, 1.0, 0.5, 1.0],
        [-1.0, 1.0, 0.5, 1.0],
    ];
    let vb_bytes: Vec<u8> = verts
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    mem.write(vb_addr, &vb_bytes);
    let ib_bytes: Vec<u8> = [0u16, 1, 2, 3, 0xFFFF, 0, 1, 2]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    mem.write(ib_addr, &ib_bytes);

    let (mut c, mut sh, mut u) = (vec![0u32; 0x400], vec![0u32; 0x400], vec![0u32; 0x400]);
    c[context::CB_COLOR0_BASE] = (target_addr >> 8) as u32;
    c[context::CB_COLOR0_INFO] = 10 << 2; // RGBA8 Unorm
    c[context::CB_COLOR0_ATTRIB2] = (w - 1) | ((h - 1) << 14);
    c[context::CB_TARGET_MASK] = 0xF;
    c[context::SPI_SHADER_COL_FORMAT] = 9;
    c[context::VGT_MULTI_PRIM_IB_RESET_EN] = 1;
    c[context::VGT_MULTI_PRIM_IB_RESET_INDX] = 0xFFFF;
    c[context::PA_CL_VPORT_XSCALE] = (w as f32 * 0.5).to_bits();
    c[context::PA_CL_VPORT_XSCALE + 1] = (w as f32 * 0.5).to_bits();
    c[context::PA_CL_VPORT_XSCALE + 2] = (h as f32 * 0.5).to_bits();
    c[context::PA_CL_VPORT_XSCALE + 3] = (h as f32 * 0.5).to_bits();
    c[context::PA_CL_VPORT_XSCALE + 4] = 1.0f32.to_bits();
    c[context::PA_CL_VPORT_XSCALE + 5] = 0.0f32.to_bits();
    sh[sh::SPI_SHADER_PGM_LO_VS] = (vs_addr >> 8) as u32;
    sh[sh::SPI_SHADER_PGM_RSRC2_VS] = 4 << 1;
    sh[sh::SPI_SHADER_USER_DATA_VS_0..sh::SPI_SHADER_USER_DATA_VS_0 + 4].copy_from_slice(&[
        vb_addr as u32,
        16 << 16,
        4,
        0,
    ]);
    sh[sh::SPI_SHADER_PGM_LO_PS] = (ps_addr >> 8) as u32;
    sh[sh::SPI_SHADER_PGM_RSRC2_PS] = 2 << 1;
    sh[sh::SPI_SHADER_USER_DATA_PS_0] = 0.0f32.to_bits();
    sh[sh::SPI_SHADER_USER_DATA_PS_0 + 1] = 1.0f32.to_bits();
    u[uconfig::VGT_PRIMITIVE_TYPE] = 5; // TriangleFan

    let state = State {
        context: &c,
        sh: &sh,
        uconfig: &u,
        color_control_written: false,
    };
    metal.draw(
        &mem,
        &state,
        crate::Draw::Indexed {
            count: 8,
            address: ib_addr,
            instances: 1,
        },
    );
    assert_eq!(metal.skipped, 0, "{:?}", metal.reasons);
    metal.flush_and_wait();
    let tex = metal
        .targets
        .get(&target_addr)
        .expect("target")
        .texture
        .clone();
    let (rw, rh, rgba) = read_rgba(&SharedTexture(tex)).expect("read_rgba");
    assert_eq!((rw, rh), (w, h));
    // Both halves of the quad (top-left and bottom-right) must be painted (0, 255, 0, 255).
    assert_eq!(&rgba[..4], &[0, 255, 0, 255]);
    assert_eq!(&rgba[rgba.len() - 4..], &[0, 255, 0, 255]);
}

#[test]
fn depth_only_draw_without_db_size_writes_depth_and_metadata_pass_is_classified() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    metal.set_wait_for_compiles(true);
    let mem = Mem::new(1 << 20);
    let (vs_addr, ps_addr, vb_addr, depth_addr) =
        (0x1_0000u64, 0x1_4000u64, 0x2_0000u64, 0x5_0000u64);
    let (w, h) = (16u32, 16u32);

    let vs: [u32; 6] = [
        0xE038_2000,
        0x8000_0100, // buffer_load_dwordx4 v[1:4], v0, s[0:3], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_08CF,
        0x0403_0201, // exp pos0 v1, v2, v3, v4 done
        0xBF81_0000, // s_endpgm
    ];
    let ps: [u32; 3] = [
        0xF800_1800,
        0x0000_0000, // exp mrt0 off, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in vs.iter().enumerate() {
        mem.write(vs_addr + 4 * i as u64, &w.to_le_bytes());
    }
    for (i, w) in ps.iter().enumerate() {
        mem.write(ps_addr + 4 * i as u64, &w.to_le_bytes());
    }
    let verts: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.25, 1.0],
        [3.0, -1.0, 0.25, 1.0],
        [-1.0, 3.0, 0.25, 1.0],
    ];
    let vb_bytes: Vec<u8> = verts
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    mem.write(vb_addr, &vb_bytes);

    let (mut c, mut sh, mut u) = (vec![0u32; 0x400], vec![0u32; 0x400], vec![0u32; 0x400]);
    c[context::DB_Z_INFO] = 3; // Depth32Float
    c[context::DB_Z_WRITE_BASE] = (depth_addr >> 8) as u32;
    // DB_DEPTH_SIZE_XY deliberately left 0: size must fall back to viewport/scissor.
    c[context::DB_DEPTH_CONTROL] = 2 | 4 | (7 << 4); // Z_ENABLE | Z_WRITE_ENABLE | Always
    c[context::PA_CL_VPORT_XSCALE] = (w as f32 * 0.5).to_bits();
    c[context::PA_CL_VPORT_XSCALE + 1] = (w as f32 * 0.5).to_bits();
    c[context::PA_CL_VPORT_XSCALE + 2] = (h as f32 * 0.5).to_bits();
    c[context::PA_CL_VPORT_XSCALE + 3] = (h as f32 * 0.5).to_bits();
    c[context::PA_CL_VPORT_XSCALE + 4] = 1.0f32.to_bits();
    c[context::PA_CL_VPORT_XSCALE + 5] = 0.0f32.to_bits();
    sh[sh::SPI_SHADER_PGM_LO_VS] = (vs_addr >> 8) as u32;
    sh[sh::SPI_SHADER_PGM_RSRC2_VS] = 4 << 1;
    sh[sh::SPI_SHADER_USER_DATA_VS_0..sh::SPI_SHADER_USER_DATA_VS_0 + 4].copy_from_slice(&[
        vb_addr as u32,
        16 << 16,
        3,
        0,
    ]);
    sh[sh::SPI_SHADER_PGM_LO_PS] = (ps_addr >> 8) as u32;
    u[uconfig::VGT_PRIMITIVE_TYPE] = 4;

    let state = State {
        context: &c,
        sh: &sh,
        uconfig: &u,
        color_control_written: false,
    };
    metal.draw(
        &mem,
        &state,
        crate::Draw::Auto {
            count: 3,
            instances: 1,
        },
    );
    assert_eq!(metal.skipped, 0, "{:?}", metal.reasons);
    assert!(metal.depths.contains_key(&depth_addr));

    // A depth resummarize metadata pass with depth test off is classified as metadata pass, not "no color target".
    c[context::DB_DEPTH_CONTROL] = 0;
    c[context::DB_RENDER_CONTROL] = 0x10; // RESUMMARIZE_ENABLE
    let meta_state = State {
        context: &c,
        sh: &sh,
        uconfig: &u,
        color_control_written: false,
    };
    metal.draw(
        &mem,
        &meta_state,
        crate::Draw::Auto {
            count: 3,
            instances: 1,
        },
    );
    assert_eq!(
        metal.reasons.get("no color target").copied().unwrap_or(0),
        0
    );
    assert_eq!(metal.reasons.get("metadata pass").copied().unwrap_or(0), 1);
    metal.flush_and_wait();
}
