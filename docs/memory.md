# Memory

What the guest expects from its address space, and what macOS lets us give it. Proposal stage; nothing is implemented.

## What the guest expects

| Fact | Source | Status |
| --- | --- | --- |
| Orbis allocates in 16 KiB pages | shadPS4 uses `GUEST_PAGE_BITS = 14` for the PS4; the PS5 radv patches use matching alignments | PS4: from an emulator. PS5: unverified |
| Executables load at `0x400000`; "system managed" `0x400000..0x7_FFFF_BFFF`, "system reserved" up to `0xF_BFFF_FFFF`, user area from `0x10_0000_0000`, GPU addresses below 2^40 | shadPS4 `address_space.cpp` | PS4 only, from an emulator. PS5 layout unknown |
| Two kinds of memory: **direct memory** (physical, allocated with `sceKernelAllocateDirectMemory`, mapped with `sceKernelMapDirectMemory`, with CPU and GPU protection bits) and **flexible memory** (ordinary virtual memory) | The PS5 radv patches call the direct-memory functions | Names observed; semantics unverified |
| TLS is reached through the FS base (`mov reg, fs:[0]`) | shadPS4 `tls.cpp` | PS4, from an emulator |

Open: whether games depend on fixed addresses (by asking for them, or by assuming where the executable lands), or only on the APIs handing back addresses in certain ranges. ET_SCE_DYNEXEC executables carry relocations, so they may load anywhere; unverified. Our own test programs and vic's PS4 can answer parts of this for the PS4.

## What macOS gives a native arm64 process

Measured on vic's M4 Pro, macOS 27.3, 2026-11-19, with `experiments/address-space-probe/run.sh`:

- Page size is 16 KiB. `mmap` and `mprotect` reject 4 KiB granularity (also seen by the CPU research pass).
- `0..4 GiB` is `__PAGEZERO`. Linking with `-Wl,-pagezero_size,0x4000` builds, but the process is killed at exec (status 137; tried by hand, not in the probe). So `0x400000` cannot be host memory in our process.
- The dyld shared cache occupies roughly `0x1_8000_0000..0x3_7000_0000`.
- A reserved guard region spans `0xF_C000_0000..0x70_0000_0000` (385 GiB): `MAP_FIXED` at `0x10_0000_0000` fails. Its purpose is unknown to us.
- `MAP_FIXED` at `0x8_0000_0000`, `0x100_0000_0000` and `0x7F00_0000_0000` worked. Our own executable and its heap land just above 4 GiB, so `0x1_0000_0000` was taken.

So "guest address = host address" does not work in-process for the PS4-style layout. Two ways out:

1. **Base offset.** Guest memory is one large reservation at a host base; host address = base + guest address. The interpreter does this for free. A JIT keeps the base in a register.
2. **HVF.** Our own stage-1 tables map guest VA directly, at any address and with 4 KiB or 16 KiB granules (`hv_vm_config_set_ipa_granule`, macOS 26+; max IPA is 40 bits on this machine). See [cpu.md](cpu.md#hypervisorframework).

Either way the guest's whole address space is one host reservation we own. That keeps the GPU side simple: GPU-visible ranges are host memory that Metal can wrap with `newBufferWithBytesNoCopy`, as 86slopper does for VRAM. Whether the same pages can be mapped into an HVF VM and wrapped by Metal at once is plausible but untested; check with an experiment before relying on it.

## Page size

If Orbis really uses 16 KiB pages, macOS's 16 KiB pages match and protections can be applied page for page. If some guest behaviour needs 4 KiB granularity (sub-16 KiB protections, guard pages), the base-offset route has to emulate it in software, while the HVF route can use a 4 KiB granule.

## The layout we use today

Our choices, in `crates/ps5slop` and `crates/hle`, until real titles tell us otherwise:

| Range | What |
| --- | --- |
| `0 .. 1 TiB` | The guest address space: one host reservation (`Memory::new`), host address = base + guest address |
| `0x40_0000 ..` | The executable, at the PS4 base shadPS4 uses (unverified for PS5) |
| `0x6_FFF0_0000 .. 0x7_0000_0000` | The main thread's stack, 1 MiB |
| `0x7_0000_0000` | The entry parameter block (argc, argv) |
| `0x5_0000_0000 ..` | Stacks of threads made by `pthread_create`, 1 MiB each |
| `0x7_0001_0000 ..` | Each thread's static TLS and thread control block (errno at the thread pointer + `0x800`) |
| `0xFF_0000_0000 ..` | The HLE trap region, one 16-byte slot per imported function |

Every mapped page is readable and writable on the host; the guest's protections live in `crates/memory`, which checks each access. This is simple and slow, and fine for the interpreter.

## Threads and memory ordering

Guest threads run on host threads (see [decisions.md](decisions.md)), so the address space is shared:

- One reservation and one region table (behind a read-write lock). Each thread holds its own `Memory` handle, a clone that caches the last region it touched and drops the cache when a generation counter shows the table changed.
- Unmapping replaces the host pages with fresh zero pages instead of making them inaccessible, so a thread racing with an unmap reads zeros rather than crashing the host.
- **x86-TSO in software:** each guest load is followed by a load barrier, each guest store preceded by a store barrier. That permits only the reordering x86 permits (an earlier store with a later load). Accesses of 1, 2, 4 and 8 bytes are single host loads and stores, so aligned ones do not tear.
- **Locked instructions** go through `Memory::atomic_update`, a compare-and-swap loop on host memory with sequentially consistent ordering. Misaligned locked accesses (which split across cache lines on x86 too) fall back to a lock that only excludes other atomic updates; a plain store from another thread could still slip in there.

**Measured on 2026-11-19** with `crates/x86/examples/bench.rs` (a five-instruction loop with one load and one store per iteration, M4 Pro): 30.0 M instructions/s before the change, 25.8 M after. With the barriers removed again it was 25.7 to 26.0 M, so the barriers cost nothing measurable here; the slowdown comes from the shared region cache's generation check. Something to revisit when the interpreter is optimised.
