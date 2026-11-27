# Memory

The guest address space: what the guest expects, what macOS allows, and how `crates/memory` provides it. How the JIT mirrors it into the VM is in [jit.md](jit.md).

## What the guest expects

| Fact | Source | Status |
| --- | --- | --- |
| Orbis allocates in 16 KiB pages | shadPS4 (`GUEST_PAGE_BITS = 14`); the PS5 radv patches use matching alignments | PS4 from an emulator; PS5 unverified |
| Executables load at `0x400000`; "system managed" `0x400000..0x7_FFFF_BFFF`, "system reserved" up to `0xF_BFFF_FFFF`, user area from `0x10_0000_0000`, GPU addresses below 2^40 | shadPS4 `address_space.cpp` | PS4 only; PS5 layout unknown |
| Two kinds of memory: **direct** (physical, `sceKernelAllocateDirectMemory` then `sceKernelMapDirectMemory`, with CPU and GPU protection bits) and **flexible** (ordinary virtual memory) | The PS5 radv patches call the direct-memory functions | Names observed; semantics partly assumed |
| TLS is reached through the FS base | shadPS4 `tls.cpp` | PS4, from an emulator |

Games do rely on fixed addresses in at least one way: GTA V maps a large streaming arena at the start of the user area with `MAP_FIXED` without reserving it first (seen in its own code). Where the PS5 kernel places maps without a fixed address is unknown.

## What macOS allows a native arm64 process

Measured on an M4 Pro, macOS 27, with `experiments/address-space-probe/run.sh`:

- Pages are 16 KiB; `mmap` and `mprotect` reject 4 KiB granularity.
- `0..4 GiB` is `__PAGEZERO`, and shrinking it gets the process killed at exec, so `0x400000` cannot be host memory at its own address.
- The dyld shared cache sits around `0x1_8000_0000..0x3_7000_0000`, and a reserved region spans `0xF_C000_0000..0x70_0000_0000`, so `MAP_FIXED` at `0x10_0000_0000` fails.

So guest address = host address is impossible in-process. Instead the guest's whole space is one host reservation at a base: host address = base + guest address. The interpreter adds the base; the JIT needs no base at all, because its own page tables map guest addresses to IPAs equal to them, and stage 2 maps those onto the reservation's pages. GPU-visible ranges are the same host memory, which Metal wraps without copying (see [gpu.md](gpu.md)).

## Layout

Our choices, until titles tell us otherwise. The constants live where noted.

- The space is 1 TiB (`ADDRESS_SPACE` in `crates/ps5slop`); GPU addresses must stay below 2^40.
- The executable loads at `0x40_0000` (`IMAGE_BASE`), the PS4 base (unverified for PS5). Modules follow one another from `0x8_0000_0000` (`MODULE_AREA` in `crates/ps5slop/src/modules.rs`).
- Stacks, TLS blocks and the entry parameter block sit below `0x8_0000_0000` (`crates/ps5slop/src/lib.rs`, `crates/hle/src/process.rs`). The main thread gets 8 MiB of stack.
- The first 4 GiB of the user area, `0x10_0000_0000..0x11_0000_0000`, is left to fixed maps. Direct memory, flexible memory, reservations and `mmap` without a fixed address are searched upward from `0x11_0000_0000` (`MAP_AREA` in `crates/hle/src/dmem.rs`). Reason: GTA V's fixed arena there replaced live memory when our maps used that range. The 4 GiB size is arbitrary beyond covering it. The GPU's guest-memory window covers the fixed part and 16 GiB of searched maps.
- The AMM area and its multimap area sit at `0x40_0000_0000` and `0x50_0000_0000`, 64 GiB each (`crates/hle/src/ampr.rs`), reserved when a title first asks for them. The reservation gives every page host memory at once (zeros); AMM map commands only change the guest protection of pieces, or zero them in place, so their pages are ordinary memory, not aliases of direct memory. Pieces of 64 MiB or more mapped as PRT get fresh pages, which `replaced_since` reports, so GPU windows over them are dropped. The GPU reaches both areas through windows near what programs point into ([gpu.md](gpu.md)).
- Memory host functions hand to the guest (`strdup`, `FILE` handles) comes from a host heap at `0x20_0000_0000` (`crates/hle/src/heap.rs`).
- `0xFE_0000_0000..0xFF_0000_0000` is never guest memory: the JIT's runtime sits there in the VM's physical space.
- `0xFF_0000_0000` up holds the HLE trap region (one 16-byte slot per imported function, `TRAP_BASE` in `crates/hle`) and then HLE variables at fixed addresses (stdio's `FILE`s and the like).

Direct memory is 12.5 GiB (`DIRECT_MEMORY_SIZE`, our choice; the PS5's figure is unknown).

## How `crates/memory` works

- **One reservation, one region table.** The table is behind a read-write lock. Each thread holds its own `Memory` handle, a clone that caches the last 8 regions it touched and drops them when the generation counter shows the table changed.
- **Guest protections are software.** Every mapped page is readable and writable on the host; the interpreter checks the guest's protection on each access. The JIT gets the same protections from its page tables.
- **Finding a free range and mapping it is one step** (`Memory::map_free`, under the region lock). Two threads must never pick the same free range.
- **Changes notify.** `memory::on_change` registers one function called on the changing thread after every change; the JIT syncs the VM there. A change log (`Memory::changes_near`) lets a guest fault print the recent changes around its address.
- **Physical memory.** Direct memory is modelled as physical memory: a shared-memory object, mapped with `MAP_FIXED` at each guest mapping (`map_physical`), so two mappings of one direct range see the same bytes, as on the console. A mapping that repeats what is already there keeps its pages. A new allocation zeroes the physical range (`zero_physical`).
- **`Memory::back`** gives every page of a range host memory without mapping it for the guest, so a GPU window can span holes.

### Unmapping never makes host pages inaccessible

A guest unmap removes the region from the table, but the host pages stay readable and writable. Reason: guest threads are host threads, and one racing with an unmap must read stale data or zeros, never crash the host.

- A new mapping under 64 MiB over pages that already exist is zeroed in place, and an unmap under 64 MiB leaves the pages alone. Reason: replacing host pages invalidates everything else that maps them, Metal's no-copy buffers and the VM's stage 2. Titles map and unmap small pieces of direct memory many times a second, and remaking GPU windows of over 1 GB each time exhausted Metal's memory.
- Larger mappings and unmaps get fresh `MAP_FIXED` pages, which is cheaper than writing zeros and gives memory back. Those ranges are logged (`take_fresh` for the VM, `replaced_since` for the GPU), and whoever mirrors those pages must map them again.

### Threads and memory ordering

Guest threads run on host threads and share the address space (see [cpu.md](cpu.md)):

- Interpreter: x86-TSO by barriers (a load barrier after each load, a store barrier before each store). Aligned accesses of 1, 2, 4 and 8 bytes are single host accesses, so they do not tear.
- Locked instructions: `Memory::atomic_update`, a sequentially consistent compare-and-swap loop on host memory, and `atomic_cas128` for `cmpxchg16b`. Misaligned locked accesses fall back to a lock that excludes only other atomic updates; a plain store from another thread could still slip in there.
- Translated code: hardware TSO in the VM, and LSE atomics on the same host words.

### Write journal and watches

`Memory` can journal writes (verify mode undoes the interpreter's writes with it) and log writes to watched ranges (`memory::watch`, used by the control socket's `watch`; the JIT protects watched pages at stage 2 so translated stores reach the interpreter).
