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
fn a_capture_records_the_memory_of_images_detiled_before_it() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mut mem = Mem::new(0x100_0000);
    mem.regions = vec![(0, 0x80_0000), (0x80_0000, 0x100_0000)];
    let at = 0x80_0000;
    mem.write(at, &[0x3f; 1024]);
    let image = Image {
        address: at,
        format: 22, // R32Float
        width: 4,
        height: 4,
        swizzle: [4, 5, 6, 7],
        tile: Some(TileMode::Linear),
        raw_tile: 0,
        kind: 9,
        base_level: 0,
        last_level: 0,
        depth: 1,
        base_array: 0,
    };
    assert!(metal.image(&mem, image, TextureKind::Tex2d).is_some());
    let recorder = crate::capture::FrameRecorder::new(None, Vec::new());
    metal.capture_starting();
    let host = recorder.recording_host(&mem);
    assert!(metal.image(&host, image, TextureKind::Tex2d).is_some());
    let page = at & !(crate::capture::PAGE_SIZE - 1);
    assert!(recorder.finish().pages.contains_key(&page));
}

#[test]
fn a_depth_layout_image_a_program_stored_is_sampled_as_stored() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(0x100_0000);
    let image = |format| Image {
        address: 0x80_0000,
        format,
        width: 64,
        height: 64,
        swizzle: [4, 0, 0, 1],
        tile: None,
        raw_tile: 0x18,
        kind: 9,
        base_level: 0,
        last_level: 0,
        depth: 1,
        base_array: 0,
    };
    let stored = metal
        .storage_image(&mem, image(0x14), TextureKind::Tex2d) // R32Uint, as a depth copy is written
        .expect("a storage image for the copy");
    let sampled = metal
        .image(&mem, image(0x16), TextureKind::Tex2d) // R32Float, as it is read
        .expect("the copy as a sampled texture");
    let parent = sampled.parentTexture();
    let same = parent.as_deref().map_or(&*sampled, |p| p) as *const _ as *const ()
        == &*stored as *const _ as *const ();
    assert!(
        same,
        "sampling must see the stored copy, not a blank texture"
    );
}

#[test]
fn a_layered_depth_target_takes_every_slice_a_program_stored() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mem = Mem::new(0x100_0000);
    let (at, n) = (0x80_0000, 64usize);
    let image = Image {
        address: at,
        format: 11, // R16Uint, as GTA V clears a cube shadow map
        width: n as u32,
        height: n as u32,
        swizzle: [4, 0, 0, 1],
        tile: None,
        raw_tile: 0x18,
        kind: 13,
        base_level: 0,
        last_level: 0,
        depth: 6,
        base_array: 0,
    };
    let stored = metal
        .storage_image(&mem, image, TextureKind::Tex2dArray)
        .expect("a layered storage image");
    let stored_words = metal
        .device
        .newBufferWithLength_options(n * n * 2, MTLResourceOptions::StorageModeShared)
        .unwrap();
    // Not 0xffff: a new depth target starts cleared to 1.0.
    unsafe {
        std::ptr::write_bytes(
            stored_words.contents().as_ptr().cast::<u8>(),
            0x12,
            n * n * 2,
        )
    };
    let size = MTLSize {
        width: n,
        height: n,
        depth: 1,
    };
    let origin = MTLOrigin { x: 0, y: 0, z: 0 };
    let cb = metal.command();
    let blit = cb.blitCommandEncoder().unwrap();
    unsafe {
        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            &stored_words, 0, n * 2, n * n * 2, size, &stored, 3, 0, origin,
        );
    }
    blit.endEncoding();
    let target = DepthTarget {
        address: at,
        format: 1,
        stencil: false,
        stencil_address: 0,
        width: n as u32,
        height: n as u32,
        slice: 3,
        slice_max: 5,
    };
    let (depth, _, _) = metal
        .depth(&target, n as u32, n as u32)
        .expect("the depth target");
    let out = metal
        .device
        .newBufferWithLength_options(n * n * 2, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let cb = metal.command();
    let blit = cb.blitCommandEncoder().unwrap();
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            &depth, 3, 0, origin, size, &out, 0, n * 2, n * n * 2,
        );
    }
    blit.endEncoding();
    metal.flush_and_wait();
    let z = unsafe { *out.contents().as_ptr().cast::<u16>().add(n * 10 + 10) };
    assert_eq!(
        z, 0x1212,
        "slice 3 of the depth target holds what was stored there"
    );
}

#[test]
fn a_capture_records_the_memory_of_windows_made_before_it() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mut mem = Mem::new(0x100_0000);
    mem.regions = vec![(0, 0x80_0000), (0x80_0000, 0x100_0000)];
    let at = 0x80_0000;
    mem.write(at, &[0x3f; 1024]);
    assert!(metal.window(&mem, at, 1024).is_some());
    let recorder = crate::capture::FrameRecorder::new(None, Vec::new());
    metal.capture_starting();
    assert!(
        metal
            .window(&recorder.recording_host(&mem), at, 1024)
            .is_some()
    );
    assert!(recorder.finish().pages.contains_key(&at));
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

// A wave64 scan carries its low half into its high half, and a radix sort ranks keys with
// LDS atomics in lane order, as GTA V's BVH builds do in groups of four waves.
#[test]
fn waves_of_a_bigger_group_keep_their_halves_in_step() {
    let Some((got, _)) = four_waves(1, false) else {
        return;
    };
    let want: Vec<u32> = (0..256)
        .map(|t| {
            let lane = t % 64;
            let carried = if lane < 32 { 1 } else { 32 };
            carried << 24 | 32 << 16 | lane << 8 | lane
        })
        .collect();
    assert_eq!(got[..256], want);
}

// Dimensions in threads leave the second group four lanes; a wave without its
// second SIMD group would wait out every exchange.
#[test]
fn a_partial_group_of_waves_runs_whole() {
    let Some((got, took)) = four_waves(260, true) else {
        return;
    };
    let ranks: Vec<u32> = got[..4].iter().map(|w| w & 0xFFFF).collect();
    assert_eq!(ranks, [0, 0x101, 0x202, 0x303]);
    assert!(took.as_millis() < 100, "the dispatch took {took:?}");
}

/// Runs a program of four wave64s per group that exchanges values across wave
/// halves, for `x` groups (or threads), and returns what each thread stored.
fn four_waves(x: u32, threads: bool) -> Option<(Vec<u32>, std::time::Duration)> {
    let metal = Metal::new()?;
    let mut mem = Mem::new(1 << 20);
    mem.area = Some((0, 1 << 20));
    let out = 0x8000u64;
    let program: &[u32] = &[
        0xD765_0002,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v2, -1, 0
        0xD766_0002,
        0x0002_04C1, // v_mbcnt_hi_u32_b32 v2, -1, v2
        0x7E02_0281, // v_mov_b32_e32 v1, 1
        0x2C08_0086, // v_lshrrev_b32_e32 v4, 6, v0
        0x3408_0882, // v_lshlrev_b32_e32 v4, 2, v4
        0x7E0A_0281, // v_mov_b32_e32 v5, 1
        0x7E10_0280, // v_mov_b32_e32 v8, 0
        0xD834_0000,
        0x0000_0804, // ds_write_b32 v4, v8
        0xD880_0000,
        0x0300_0504, // ds_add_rtn_u32 v3, v4, v5
        0x92FE_A0A0, // s_bfm_b64 exec, 32, 32
        0xD760_0008,
        0x0001_3F02, // v_readlane_b32 s8, v2, 31
        0x4A02_0208, // v_add_nc_u32_e32 v1, s8, v1
        0xBEFE_04C1, // s_mov_b64 exec, -1
        0xD760_0009,
        0x0001_7F01, // v_readlane_b32 s9, v1, 63
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x340C_0298, // v_lshlrev_b32_e32 v6, 24, v1
        0x8F09_9009, // s_lshl_b32 s9, s9, 16
        0x380C_0C09, // v_or_b32_e32 v6, s9, v6
        0x3406_0688, // v_lshlrev_b32_e32 v3, 8, v3
        0x380C_0D03, // v_or_b32_e32 v6, v3, v6
        0x380C_0D02, // v_or_b32_e32 v6, v2, v6
        0x340E_0082, // v_lshlrev_b32_e32 v7, 2, v0
        0xE070_1000,
        0x8001_0607, // buffer_store_dword v6, v7, s[4:7], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    for (i, w) in program.iter().enumerate() {
        mem.write(0x1000 + 4 * i as u64, &w.to_le_bytes());
    }
    let store = [out as u32, (out >> 32) as u32, 4096, 0x5204];
    let mut packets = [
        compute_setup(0x1000, &[buffer(0), store].concat()),
        DISPATCH_ONE.to_vec(),
    ]
    .concat();
    // COMPUTE_NUM_THREAD_X: four wave64s.
    packets[9] = 256;
    let n = packets.len();
    packets[n - 4] = x;
    if threads {
        packets[n - 1] |= 0x20;
    }
    let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
    mem.write(0x2000, &bytes);
    let mut p = crate::pm4::Processor::default();
    p.backend = Some(Box::new(metal));
    p.run(&mem, 0x2000, packets.len() as u32).unwrap();
    p.finish(&mem);
    let got = (0..512).map(|i| word(&mem, out + 4 * i)).collect();
    // Timed again once the program is compiled: a cold compile alone can take longer than the limit.
    let start = std::time::Instant::now();
    p.run(&mem, 0x2000, packets.len() as u32).unwrap();
    p.finish(&mem);
    Some((got, start.elapsed()))
}

#[test]
fn a_target_mask_picks_export_channels_on_a_bgra_target() {
    let target = regs::ColorFormat {
        layout: 10,
        number: 0,
        swap: 1,
    };
    assert_eq!(color_format(target), Some(MTLPixelFormat::BGRA8Unorm));
    assert_eq!(write_mask(0b0100), MTLColorWriteMask::Blue.0);
    assert_eq!(
        write_mask(0b1001),
        (MTLColorWriteMask::Red | MTLColorWriteMask::Alpha).0
    );
}

#[test]
fn a_program_in_a_mapping_that_runs_past_the_gpu_area_gets_a_window() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    let mut mem = Mem::new(0x100_0000);
    mem.area = Some((0x10_0000, 0x40_0000));
    mem.regions = vec![(0x20_0000, 0x60_0000)];
    let memory = metal
        .guest_window(&mem, &[], 0x30_0000)
        .expect("a window for the program");
    assert_eq!(memory.win[0], 0x10_0000);
    assert!(memory.win[1] > 0x30_0000);
}

#[test]
fn a_masked_target_the_program_exports_to_keeps_its_attachment() {
    let hair: Vec<_> = attached_targets(0x4307, 0x4444).collect();
    assert_eq!(hair, [0, 1, 2, 3]);
    let unexported: Vec<_> = attached_targets(0x4307, 0x4404).collect();
    assert_eq!(unexported, [0, 2, 3]);
}

#[test]
fn a_masked_export_after_the_last_written_target_is_not_attached() {
    let depth_only: Vec<_> = attached_targets(0, 0x1).collect();
    assert!(depth_only.is_empty());
    let trailing: Vec<_> = attached_targets(0x7, 0x44).collect();
    assert_eq!(trailing, [0]);
}
