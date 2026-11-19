//! Host implementations of system library functions. Each import binds to
//! its own address in a trap region; when guest code calls it, the
//! interpreter stops there and [`Hle::call`] runs the host function and
//! returns to the caller. Imports we do not implement still get an address,
//! so a call reports which function is missing instead of jumping to 0.

use std::collections::HashMap;

use loader::dynamic::Import;
use loader::image::Resolver;
use loader::nid::nid;
use memory::{Fault, MapError, Memory, PAGE_SIZE, Prot};
use x86::Cpu;
use x86::cpu::reg::*;

mod dmem;
mod equeue;
pub mod errno;
mod file;
mod heap;
mod kernel;
mod libc;
mod pad;
mod posix;
mod process;
mod pthread;
mod sem;
mod services;
mod sysmodule;
mod thread;
mod vfs;
mod videoout;

pub use pad::PadInput;
pub use process::{Kernel, Thread, ThreadError, TlsTemplate};
pub use vfs::Vfs;
pub use videoout::Frame;

/// Where the trap region starts. Our choice, below 2^40 and away from where
/// PS4 titles put things (see `docs/memory.md`).
pub const TRAP_BASE: u64 = 0x00ff_0000_0000;
/// The library name of functions only ps5slop itself calls.
pub const INTERNAL: &str = "ps5slop";
/// The teardown callback handed to the entry point in RSI. Start-up code
/// registers it with `atexit`, so it returns.
pub const LOADER_TEARDOWN: &str = "loader_teardown";

/// Where host-provided variables live (readable and writable). The first
/// [`FIXED_DATA`] bytes hold objects at fixed addresses (stdio's FILEs).
pub const DATA_BASE: u64 = 0x00ff_8000_0000;
pub const FIXED_DATA: u64 = 0x1000;

/// The internal function a guest callback returns to.
pub const CALLBACK_RETURN: &str = "callback_return";

/// Bytes of trap region per function.
const STUB_SIZE: u64 = 16;
const MAX_STUBS: u64 = 0x10_0000;

/// What a host function asks the caller to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Return to the guest with this value in RAX.
    Return(u64),
    /// Continue the guest as the host function left the CPU (for example
    /// after setting it up to call a guest function).
    Resume,
    /// The calling thread ends.
    ThreadExit,
    /// The process exits with this status.
    Exit(i32),
}

#[derive(Debug)]
pub enum CallError {
    /// An address in the trap region that is not the start of a stub.
    NotAStub(u64),
    /// A call to a function we do not implement.
    Unimplemented {
        nid: String,
        name: Option<String>,
        library: String,
    },
    /// A guest pointer argument that faults.
    Memory(Fault),
    /// A guest function called from a host function faulted.
    GuestFault { fault: x86::CpuFault, rip: u64 },
    /// The process exited while a host function was calling guest code.
    Exited,
    /// The guest asked to stop because something went wrong (for example a
    /// failed stack check).
    Abort(&'static str),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::NotAStub(a) => write!(f, "{a:#x} is not a host function"),
            CallError::Unimplemented {
                nid,
                name: Some(n),
                library,
            } => {
                write!(f, "unimplemented {library} function {n} ({nid})")
            }
            CallError::Unimplemented {
                nid,
                name: None,
                library,
            } => {
                write!(f, "unimplemented {library} function with NID {nid}")
            }
            CallError::Memory(e) => write!(f, "bad guest pointer: {e}"),
            CallError::GuestFault { fault, rip } => {
                write!(f, "guest callback faulted ({fault:?}) at {rip:#x}")
            }
            CallError::Exited => write!(f, "the process exited during a guest callback"),
            CallError::Abort(why) => write!(f, "guest aborted: {why}"),
        }
    }
}

impl std::error::Error for CallError {}

impl From<Fault> for CallError {
    fn from(f: Fault) -> Self {
        CallError::Memory(f)
    }
}

/// What a host function sees: the calling thread's state and the process.
pub struct Env<'a> {
    pub cpu: &'a mut Cpu,
    pub mem: &'a Memory,
    pub kernel: &'a Kernel,
    pub thread: Thread,
}

impl Env<'_> {
    /// Sets the calling thread's errno.
    pub fn set_errno(&self, errno: i32) -> Result<(), CallError> {
        self.mem
            .write(self.thread.errno_addr(), &errno.to_le_bytes())?;
        Ok(())
    }

    /// Fails a POSIX-style call: errno is set and the call returns -1.
    pub fn fail(&self, errno: i32) -> Result<Outcome, CallError> {
        self.set_errno(errno)?;
        Ok(Outcome::Return(-1i64 as u64))
    }

    /// Calls the guest function at `addr` with integer arguments `args` (at
    /// most six) and returns its RAX. The guest runs on this thread, and the
    /// caller's CPU state is restored afterwards.
    pub fn call_guest(&mut self, addr: u64, args: &[u64]) -> Result<u64, CallError> {
        const ORDER: [usize; 6] = [RDI, RSI, RDX, RCX, R8, R9];
        let saved = self.cpu.clone();
        let back = self.kernel.hle.internal(CALLBACK_RETURN);
        // Skip the red zone, align, and push the return address as a call would.
        let sp = ((saved.regs[RSP] - 128) & !15) - 8;
        self.mem.write_u64(sp, back)?;
        self.cpu.regs[RSP] = sp;
        self.cpu.rip = addr;
        for (i, &a) in args.iter().enumerate() {
            self.cpu.regs[ORDER[i]] = a;
        }
        let result = loop {
            if self.kernel.has_exited() {
                break Err(CallError::Exited);
            }
            match self.kernel.execute(self.cpu, self.mem) {
                x86::Exit::Continue | x86::Exit::Limit => {}
                x86::Exit::Trap(t) if t == back => break Ok(self.cpu.regs[RAX]),
                x86::Exit::Trap(t) => match self.kernel.hle.call(t, self) {
                    Ok(Outcome::Return(_) | Outcome::Resume) => {}
                    Ok(Outcome::Exit(status)) => {
                        self.kernel.exit(Ok(status));
                        break Err(CallError::Exited);
                    }
                    Ok(Outcome::ThreadExit) => {
                        break Err(CallError::Abort("a thread ended inside a callback"));
                    }
                    Err(e) => break Err(e),
                },
                x86::Exit::Fault { rip, kind } => {
                    break Err(CallError::GuestFault { fault: kind, rip });
                }
                x86::Exit::Syscall => break Err(CallError::Abort("raw syscall inside a callback")),
            }
        };
        *self.cpu = saved;
        result
    }

    /// Integer argument `n` in the SysV AMD64 convention.
    pub fn arg(&self, n: usize) -> u64 {
        const ORDER: [usize; 6] = [RDI, RSI, RDX, RCX, R8, R9];
        self.cpu.regs[ORDER[n]]
    }
}

pub type HostFn = fn(&mut Env) -> Result<Outcome, CallError>;

/// A variable we provide, by library and name, with its initial contents.
pub struct Variable {
    pub library: &'static str,
    pub name: &'static str,
    pub init: &'static [u8],
}

/// A function we implement, by library and name.
pub struct Function {
    pub library: &'static str,
    pub name: &'static str,
    pub func: HostFn,
}

struct Stub {
    nid: String,
    library: String,
    func: Option<&'static Function>,
}

pub struct Hle {
    by_nid: HashMap<(String, String), &'static Function>,
    stubs: Vec<Stub>,
    stub_index: HashMap<(String, String), usize>,
    variables: HashMap<(String, String), &'static Variable>,
    /// Variables handed out so far: address and contents.
    data: Vec<(u64, &'static Variable)>,
    data_end: u64,
    /// Names for NIDs: ours, from every library, and an optional local list.
    names: HashMap<String, String>,
}

impl Default for Hle {
    fn default() -> Self {
        Self::new()
    }
}

impl Hle {
    pub fn new() -> Self {
        let by_nid = functions()
            .map(|f| ((nid(f.name), f.library.to_owned()), f))
            .collect();
        let variables = kernel::VARIABLES
            .iter()
            .chain(libc::VARIABLES)
            .map(|v| ((nid(v.name), v.library.to_owned()), v))
            .collect();
        let mut hle = Self {
            by_nid,
            stubs: Vec::new(),
            stub_index: HashMap::new(),
            variables,
            data: Vec::new(),
            data_end: DATA_BASE + FIXED_DATA,
            names: names(),
        };
        let internal: Vec<&str> = hle
            .by_nid
            .iter()
            .filter(|(k, _)| k.1 == INTERNAL)
            .map(|(_, f)| f.name)
            .collect();
        for name in internal {
            hle.stub_by_name(INTERNAL, name);
        }
        hle
    }

    /// The address `nid` in `library` binds to: a variable we provide, or a
    /// function stub, made on first use.
    pub fn stub(&mut self, nid: &str, library: &str) -> u64 {
        let key = (nid.to_owned(), library.to_owned());
        if let Some(&v) = self.variables.get(&key) {
            if let Some(&(addr, _)) = self.data.iter().find(|(_, d)| std::ptr::eq(*d, v)) {
                return addr;
            }
            let addr = self.data_end;
            self.data_end = (addr + v.init.len() as u64).next_multiple_of(16);
            self.data.push((addr, v));
            return addr;
        }
        let index = match self.stub_index.get(&key) {
            Some(&i) => i,
            None => {
                let i = self.stubs.len();
                self.stubs.push(Stub {
                    nid: key.0.clone(),
                    library: key.1.clone(),
                    func: self.by_nid.get(&key).copied(),
                });
                self.stub_index.insert(key, i);
                i
            }
        };
        assert!((index as u64) < MAX_STUBS, "too many imported functions");
        TRAP_BASE + index as u64 * STUB_SIZE
    }

    /// The stub address of one of our internal functions, which exist from
    /// the start.
    pub fn internal(&self, name: &str) -> u64 {
        let index = self.stub_index[&(nid(name), INTERNAL.to_owned())];
        TRAP_BASE + index as u64 * STUB_SIZE
    }

    /// The stub address an imported function already has, by library and name.
    pub fn bound(&self, library: &str, name: &str) -> Option<u64> {
        let index = *self.stub_index.get(&(nid(name), library.to_owned()))?;
        Some(TRAP_BASE + index as u64 * STUB_SIZE)
    }

    /// The stub address of a function by name.
    pub fn stub_by_name(&mut self, library: &str, name: &str) -> u64 {
        self.stub(&nid(name), library)
    }

    /// A name for `nid`, if we know one: a function we implement (in any
    /// library), or one from the `PS5SLOP_NAMES` list.
    pub fn name_of(&self, nid: &str) -> Option<&str> {
        self.names.get(nid).map(String::as_str)
    }

    /// `(library, NID, name if known)` of every stub with no host function
    /// behind it.
    pub fn unimplemented(&self) -> Vec<(String, String, Option<String>)> {
        self.stubs
            .iter()
            .filter(|s| s.func.is_none())
            .map(|s| {
                let name = self.name_of(&s.nid).map(str::to_owned);
                (s.library.clone(), s.nid.clone(), name)
            })
            .collect()
    }

    /// Maps the trap region that holds every stub handed out so far.
    pub fn map(&self, mem: &Memory) -> Result<(), MapError> {
        let len = (self.stubs.len() as u64 * STUB_SIZE)
            .next_multiple_of(PAGE_SIZE)
            .max(PAGE_SIZE);
        mem.map_trap(TRAP_BASE, len, "hle")?;
        let data_len = (self.data_end - DATA_BASE)
            .next_multiple_of(PAGE_SIZE)
            .max(PAGE_SIZE);
        mem.map(DATA_BASE, data_len, Prot::RW, "hle data")?;
        for (addr, v) in &self.data {
            mem.write(*addr, v.init).expect("hle data was just mapped");
        }
        Ok(())
    }

    /// Runs the host function whose stub is at `addr`, as if the guest had
    /// called it: on [`Outcome::Return`], RAX holds the result and RIP the
    /// return address popped from the stack.
    pub fn call(&self, addr: u64, env: &mut Env) -> Result<Outcome, CallError> {
        let index = addr
            .checked_sub(TRAP_BASE)
            .filter(|o| o % STUB_SIZE == 0)
            .map(|o| (o / STUB_SIZE) as usize);
        let stub = index
            .and_then(|i| self.stubs.get(i))
            .ok_or(CallError::NotAStub(addr))?;
        let f = stub.func.ok_or_else(|| CallError::Unimplemented {
            nid: stub.nid.clone(),
            name: self.name_of(&stub.nid).map(str::to_owned),
            library: stub.library.clone(),
        })?;
        count_call(f);
        let out = (f.func)(env)?;
        if let Outcome::Return(v) = out {
            let sp = env.cpu.regs[RSP];
            env.cpu.rip = env.mem.read_u64(sp)?;
            env.cpu.regs[RSP] = sp + 8;
            env.cpu.regs[RAX] = v;
        }
        Ok(out)
    }
}

/// Every function we implement.
fn functions() -> impl Iterator<Item = &'static Function> {
    kernel::FUNCTIONS
        .iter()
        .chain(sysmodule::FUNCTIONS)
        .chain(pthread::FUNCTIONS)
        .chain(thread::FUNCTIONS)
        .chain(sem::FUNCTIONS)
        .chain(dmem::FUNCTIONS)
        .chain(equeue::FUNCTIONS)
        .chain(posix::FUNCTIONS)
        .chain(file::FUNCTIONS)
        .chain(libc::functions())
        .chain(services::FUNCTIONS)
        .chain(pad::FUNCTIONS)
        .chain(videoout::FUNCTIONS)
}

/// Names for NIDs: everything we implement, and the names listed (one per
/// line) in the file `PS5SLOP_NAMES` points to, hashed here.
fn names() -> HashMap<String, String> {
    let mut names = HashMap::new();
    let ours = functions().map(|f| f.name).chain(
        kernel::VARIABLES
            .iter()
            .chain(libc::VARIABLES)
            .map(|v| v.name),
    );
    for name in ours {
        names.insert(nid(name), name.to_owned());
    }
    if let Some(path) = std::env::var_os("PS5SLOP_NAMES") {
        match std::fs::read_to_string(&path) {
            Ok(list) => {
                for name in list.lines().map(str::trim).filter(|n| !n.is_empty()) {
                    names.entry(nid(name)).or_insert_with(|| name.to_owned());
                }
            }
            Err(e) => eprintln!("ps5slop: cannot read {}: {e}", path.to_string_lossy()),
        }
    }
    names
}

/// Host calls by function, counted in statistics mode (`PS5SLOP_JIT_STATS`).
fn calls() -> Option<&'static std::sync::Mutex<HashMap<&'static str, u64>>> {
    static CALLS: std::sync::OnceLock<Option<std::sync::Mutex<HashMap<&'static str, u64>>>> =
        std::sync::OnceLock::new();
    CALLS
        .get_or_init(|| std::env::var_os("PS5SLOP_JIT_STATS").map(|_| Default::default()))
        .as_ref()
}

fn count_call(f: &Function) {
    if let Some(c) = calls() {
        *c.lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(f.name)
            .or_default() += 1;
    }
}

/// Prints the host calls statistics mode counted, most frequent first.
pub fn report_calls() {
    let Some(c) = calls() else { return };
    let c = c.lock().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<_> = c.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("  host calls:");
    for (name, n) in v.iter().take(20) {
        eprintln!("    {n:>12}  {name}");
    }
}

impl Resolver for Hle {
    fn resolve(&mut self, import: &Import<'_>) -> Option<u64> {
        Some(self.stub(import.nid, &import.library.name))
    }
}

#[cfg(test)]
mod tests {
    use memory::Prot;

    use super::*;

    /// Output written by the guest, shared with the test.
    #[derive(Clone, Default)]
    struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A kernel around `hle` with a stack and a data page, and a CPU whose
    /// stack holds the return address 0x40_1234.
    fn guest(hle: Hle) -> (std::sync::Arc<Kernel>, Cpu, Captured) {
        guest_in(hle, None)
    }

    /// [`guest`], with `app0` as the title folder.
    fn guest_in(
        hle: Hle,
        app0: Option<std::path::PathBuf>,
    ) -> (std::sync::Arc<Kernel>, Cpu, Captured) {
        let mem = Memory::new(1 << 40).unwrap();
        mem.map(0x10_0000, PAGE_SIZE, Prot::RW, "stack").unwrap();
        mem.map(0x20_0000, PAGE_SIZE, Prot::RW, "data").unwrap();
        hle.map(&mem).unwrap();
        let mut cpu = Cpu::default();
        cpu.regs[RSP] = 0x10_0000 + PAGE_SIZE - 8;
        mem.write_u64(cpu.regs[RSP], 0x40_1234).unwrap();
        let out = Captured::default();
        let kernel = Kernel::new(
            hle,
            mem,
            TlsTemplate::default(),
            Box::new(out.clone()),
            Box::new(std::io::sink()),
            Vfs::new(app0, None),
        );
        cpu.fs_base = kernel.new_tls(&kernel.memory()).unwrap();
        (kernel, cpu, out)
    }

    fn call(kernel: &Kernel, stub: u64, cpu: &mut Cpu) -> Result<Outcome, CallError> {
        let mem = kernel.memory();
        let thread = Thread { tcb: cpu.fs_base };
        let mut env = Env {
            cpu,
            mem: &mem,
            kernel,
            thread,
        };
        kernel.hle.call(stub, &mut env)
    }

    #[test]
    fn write_copies_guest_bytes_to_stdout_and_returns() {
        let mut hle = Hle::new();
        let stub = hle.stub_by_name("libkernel", "write");
        let (kernel, mut cpu, out) = guest(hle);
        kernel.memory().write(0x20_0000, b"hi\n").unwrap();
        cpu.regs[RDI] = 1;
        cpu.regs[RSI] = 0x20_0000;
        cpu.regs[RDX] = 3;
        assert_eq!(call(&kernel, stub, &mut cpu).unwrap(), Outcome::Return(3));
        assert_eq!(*out.0.lock().unwrap(), b"hi\n");
        assert_eq!(
            (cpu.rip, cpu.regs[RAX], cpu.regs[RSP]),
            (0x40_1234, 3, 0x10_0000 + PAGE_SIZE)
        );
    }

    #[test]
    fn exit_reports_the_status() {
        let mut hle = Hle::new();
        let stub = hle.stub_by_name("libkernel", "exit");
        let (kernel, mut cpu, _) = guest(hle);
        cpu.regs[RDI] = 7;
        assert_eq!(call(&kernel, stub, &mut cpu).unwrap(), Outcome::Exit(7));
    }

    #[test]
    fn unknown_functions_get_a_stub_that_reports_them() {
        let mut hle = Hle::new();
        let stub = hle.stub("AAAAAAAAAAA", "libSceFoo");
        assert_eq!(
            hle.unimplemented(),
            [("libSceFoo".to_owned(), "AAAAAAAAAAA".to_owned(), None)]
        );
        let (kernel, mut cpu, _) = guest(hle);
        assert!(matches!(
            call(&kernel, stub, &mut cpu),
            Err(CallError::Unimplemented { .. })
        ));
    }

    #[test]
    fn variables_bind_to_readable_data() {
        let mut hle = Hle::new();
        let guard = hle.stub_by_name("libkernel", "__stack_chk_guard");
        assert_eq!(hle.stub_by_name("libkernel", "__stack_chk_guard"), guard);
        let (kernel, _, _) = guest(hle);
        assert_ne!(kernel.memory().read_u64(guard).unwrap(), 0);
    }

    #[test]
    fn failing_calls_set_the_calling_threads_errno() {
        let mut hle = Hle::new();
        let ioctl = hle.stub_by_name("libkernel", "ioctl");
        let error = hle.stub_by_name("libkernel", "__error");
        let (kernel, mut cpu, _) = guest(hle);
        assert_eq!(
            call(&kernel, ioctl, &mut cpu).unwrap(),
            Outcome::Return(u64::MAX)
        );
        cpu.regs[RSP] -= 8;
        call(&kernel, error, &mut cpu).unwrap();
        let mut b = [0u8; 4];
        kernel.memory().read(cpu.regs[RAX], &mut b).unwrap();
        assert_eq!(i32::from_le_bytes(b), errno::ENOTTY);
    }

    /// Calls `name` with `args` as the thread whose pointer is `tcb`.
    fn libkernel(kernel: &Kernel, name: &str, tcb: u64, args: &[u64]) -> u64 {
        const ORDER: [usize; 6] = [RDI, RSI, RDX, RCX, R8, R9];
        let stub = kernel.hle.stub_index[&(nid(name), "libkernel".to_owned())];
        let mut cpu = Cpu {
            fs_base: tcb,
            ..Cpu::default()
        };
        cpu.regs[RSP] = 0x10_0000 + PAGE_SIZE - 8;
        for (i, &a) in args.iter().enumerate() {
            cpu.regs[ORDER[i]] = a;
        }
        let mem = kernel.memory();
        mem.write_u64(cpu.regs[RSP], 0x40_1234).unwrap();
        let thread = Thread { tcb };
        let mut env = Env {
            cpu: &mut cpu,
            mem: &mem,
            kernel,
            thread,
        };
        kernel
            .hle
            .call(TRAP_BASE + stub as u64 * STUB_SIZE, &mut env)
            .unwrap();
        cpu.regs[RAX]
    }

    fn with_pthreads() -> std::sync::Arc<Kernel> {
        let mut hle = Hle::new();
        for name in [
            "pthread_mutex_lock",
            "pthread_mutex_unlock",
            "pthread_mutex_trylock",
            "pthread_mutex_init",
            "pthread_mutexattr_init",
            "pthread_mutexattr_settype",
            "pthread_key_create",
            "pthread_setspecific",
            "pthread_getspecific",
        ] {
            hle.stub_by_name("libkernel", name);
        }
        guest(hle).0
    }

    #[test]
    fn default_mutexes_check_for_errors() {
        let k = with_pthreads();
        let m = 0x20_0000;
        assert_eq!(libkernel(&k, "pthread_mutex_lock", 1, &[m]), 0);
        assert_eq!(
            libkernel(&k, "pthread_mutex_lock", 1, &[m]) as i32,
            errno::EDEADLK
        );
        assert_eq!(
            libkernel(&k, "pthread_mutex_trylock", 2, &[m]) as i32,
            errno::EBUSY
        );
        assert_eq!(
            libkernel(&k, "pthread_mutex_unlock", 2, &[m]) as i32,
            errno::EPERM
        );
        assert_eq!(libkernel(&k, "pthread_mutex_unlock", 1, &[m]), 0);
        assert_eq!(libkernel(&k, "pthread_mutex_trylock", 2, &[m]), 0);
    }

    #[test]
    fn recursive_mutexes_count_their_locks() {
        let k = with_pthreads();
        let (attr, m) = (0x20_0000, 0x20_0010);
        libkernel(&k, "pthread_mutexattr_init", 1, &[attr]);
        assert_eq!(libkernel(&k, "pthread_mutexattr_settype", 1, &[attr, 2]), 0);
        libkernel(&k, "pthread_mutex_init", 1, &[m, attr]);
        assert_eq!(libkernel(&k, "pthread_mutex_lock", 1, &[m]), 0);
        assert_eq!(libkernel(&k, "pthread_mutex_lock", 1, &[m]), 0);
        assert_eq!(libkernel(&k, "pthread_mutex_unlock", 1, &[m]), 0);
        assert_eq!(
            libkernel(&k, "pthread_mutex_trylock", 2, &[m]) as i32,
            errno::EBUSY
        );
        assert_eq!(libkernel(&k, "pthread_mutex_unlock", 1, &[m]), 0);
        assert_eq!(libkernel(&k, "pthread_mutex_trylock", 2, &[m]), 0);
    }

    #[test]
    fn thread_specific_values_are_per_thread() {
        let k = with_pthreads();
        let key_at = 0x20_0000;
        libkernel(&k, "pthread_key_create", 1, &[key_at, 0]);
        let mut b = [0u8; 4];
        k.memory().read(key_at, &mut b).unwrap();
        let key = u32::from_le_bytes(b) as u64;
        libkernel(&k, "pthread_setspecific", 1, &[key, 0xaaaa]);
        libkernel(&k, "pthread_setspecific", 2, &[key, 0xbbbb]);
        assert_eq!(libkernel(&k, "pthread_getspecific", 1, &[key]), 0xaaaa);
        assert_eq!(libkernel(&k, "pthread_getspecific", 2, &[key]), 0xbbbb);
        assert_eq!(libkernel(&k, "pthread_getspecific", 3, &[key]), 0);
    }

    #[test]
    fn semaphores_count_posts_and_refuse_to_go_below_zero() {
        let mut hle = Hle::new();
        for name in [
            "sem_init",
            "sem_trywait",
            "sem_post",
            "sem_getvalue",
            "__error",
        ] {
            hle.stub_by_name("libkernel", name);
        }
        let (k, cpu, _) = guest(hle);
        let me = cpu.fs_base;
        let (sem, value) = (0x20_0000, 0x20_0100);
        assert_eq!(libkernel(&k, "sem_init", me, &[sem, 0, 1]), 0);
        assert_eq!(libkernel(&k, "sem_trywait", me, &[sem]), 0);
        assert_eq!(libkernel(&k, "sem_trywait", me, &[sem]), u64::MAX);
        assert_eq!(libkernel(&k, "sem_post", me, &[sem]), 0);
        assert_eq!(libkernel(&k, "sem_post", me, &[sem]), 0);
        libkernel(&k, "sem_getvalue", me, &[sem, value]);
        let mut b = [0u8; 4];
        k.memory().read(value, &mut b).unwrap();
        assert_eq!(u32::from_le_bytes(b), 2);
    }

    #[test]
    fn files_in_app0_open_read_and_close_through_descriptors() {
        let dir = std::env::temp_dir().join(format!("ps5slop-fd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("label.txt"), b"hello").unwrap();
        let mut hle = Hle::new();
        for name in ["open", "read", "close", "sceKernelOpen", "sceKernelMkdir"] {
            hle.stub_by_name("libkernel", name);
        }
        let (k, cpu, _) = guest_in(hle, Some(dir.clone()));
        let me = cpu.fs_base;
        let mem = k.memory();
        let (path, buf) = (0x20_0000, 0x20_0100);
        mem.write(path, b"/app0/label.txt\0").unwrap();
        let fd = libkernel(&k, "open", me, &[path, 0]);
        assert_eq!(fd, 3);
        assert_eq!(libkernel(&k, "read", me, &[fd, buf, 64]), 5);
        assert_eq!(mem.read_cstr(buf, 5).unwrap(), b"hello");
        assert_eq!(libkernel(&k, "read", me, &[fd, buf, 64]), 0);
        assert_eq!(libkernel(&k, "close", me, &[fd]), 0);
        assert_eq!(libkernel(&k, "close", me, &[fd]), u64::MAX);
        assert_eq!(
            libkernel(&k, "sceKernelOpen", me, &[path, 1, 0o666]),
            0x8002_0000 + errno::EROFS as u64
        );
        mem.write(path, b"/data/x\0").unwrap();
        assert_eq!(
            libkernel(&k, "sceKernelMkdir", me, &[path, 0o777]),
            0x8002_0000 + errno::ENOENT as u64
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn host_functions_can_call_guest_functions() {
        let (kernel, mut cpu, _) = guest(Hle::new());
        let mem = kernel.memory();
        mem.map(0x30_0000, PAGE_SIZE, memory::Prot::RX, "code")
            .unwrap();
        mem.write_host(0x30_0000, &[0x48, 0x8d, 0x04, 0x37, 0xc3])
            .unwrap();
        cpu.regs[RBX] = 0x1234;
        let before = cpu.clone();
        let thread = Thread { tcb: cpu.fs_base };
        let mut env = Env {
            cpu: &mut cpu,
            mem: &mem,
            kernel: &kernel,
            thread,
        };
        assert_eq!(env.call_guest(0x30_0000, &[2, 3]).unwrap(), 5);
        assert_eq!(cpu, before);
    }

    #[test]
    fn snprintf_reads_variadic_integer_and_float_arguments() {
        let mut hle = Hle::new();
        let stub = hle.stub_by_name("libSceLibcInternal", "snprintf");
        let (kernel, mut cpu, _) = guest(hle);
        let mem = kernel.memory();
        let (buf, fmt, name) = (0x20_0000, 0x20_0100, 0x20_0200);
        mem.write(fmt, b"%s=%04d %.2f%%\0").unwrap();
        mem.write(name, b"x\0").unwrap();
        cpu.regs[RDI] = buf;
        cpu.regs[RSI] = 9;
        cpu.regs[RDX] = fmt;
        cpu.regs[RCX] = name;
        cpu.regs[R8] = -42i64 as u64;
        cpu.xmm[0][0] = 1.5f64.to_bits();
        assert_eq!(call(&kernel, stub, &mut cpu).unwrap(), Outcome::Return(12));
        assert_eq!(mem.read_cstr(buf, 64).unwrap(), b"x=-042 1");
    }

    #[test]
    fn the_same_import_binds_to_the_same_stub() {
        let mut hle = Hle::new();
        let a = hle.stub_by_name("libkernel", "write");
        let b = hle.stub_by_name("libkernel", "exit");
        assert_ne!(a, b);
        assert_eq!(hle.stub_by_name("libkernel", "write"), a);
    }

    #[test]
    fn bad_buffers_are_errors_not_crashes() {
        let mut hle = Hle::new();
        let stub = hle.stub_by_name("libkernel", "write");
        let (kernel, mut cpu, _) = guest(hle);
        cpu.regs[RDI] = 1;
        cpu.regs[RSI] = 0x9999_0000;
        cpu.regs[RDX] = 4;
        assert!(matches!(
            call(&kernel, stub, &mut cpu),
            Err(CallError::Memory(_))
        ));
    }
}
