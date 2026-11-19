//! Host implementations of system library functions. Each import binds to
//! its own address in a trap region; when guest code calls it, the
//! interpreter stops there and [`Hle::call`] runs the host function and
//! returns to the caller. Imports we do not implement still get an address,
//! so a call reports which function is missing instead of jumping to 0.

use std::collections::HashMap;
use std::io::Write;

use loader::dynamic::Import;
use loader::image::Resolver;
use loader::nid::nid;
use memory::{Fault, MapError, Memory, PAGE_SIZE, Prot};
use x86::Cpu;
use x86::cpu::reg::*;

mod kernel;

/// Where the trap region starts. Our choice, below 2^40 and away from where
/// PS4 titles put things (see `docs/memory.md`).
pub const TRAP_BASE: u64 = 0x00ff_0000_0000;
/// The library name of functions only ps5slop itself calls.
pub const INTERNAL: &str = "ps5slop";
/// The exit function handed to the entry point in RSI: `exit(status)`.
pub const PROGRAM_EXIT: &str = "program_exit";

/// Where host-provided variables live (readable and writable).
pub const DATA_BASE: u64 = 0x00ff_8000_0000;

/// Bytes of trap region per function.
const STUB_SIZE: u64 = 16;
const MAX_STUBS: u64 = 0x10_0000;

/// What a host function asks the caller to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Return to the guest with this value in RAX.
    Return(u64),
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
        name: Option<&'static str>,
        library: String,
    },
    /// A guest pointer argument that faults.
    Memory(Fault),
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

/// What a host function sees: the guest state and its arguments.
pub struct Env<'a> {
    pub cpu: &'a mut Cpu,
    pub mem: &'a mut Memory,
    pub stdout: &'a mut dyn Write,
    pub stderr: &'a mut dyn Write,
}

impl Env<'_> {
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
}

impl Default for Hle {
    fn default() -> Self {
        Self::new()
    }
}

impl Hle {
    pub fn new() -> Self {
        let by_nid = kernel::FUNCTIONS
            .iter()
            .map(|f| ((nid(f.name), f.library.to_owned()), f))
            .collect();
        let variables = kernel::VARIABLES
            .iter()
            .map(|v| ((nid(v.name), v.library.to_owned()), v))
            .collect();
        Self {
            by_nid,
            stubs: Vec::new(),
            stub_index: HashMap::new(),
            variables,
            data: Vec::new(),
            data_end: DATA_BASE,
        }
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

    /// The stub address of a function by name.
    pub fn stub_by_name(&mut self, library: &str, name: &str) -> u64 {
        self.stub(&nid(name), library)
    }

    /// `(library, NID)` of every stub with no host function behind it.
    pub fn unimplemented(&self) -> Vec<(String, String)> {
        self.stubs
            .iter()
            .filter(|s| s.func.is_none())
            .map(|s| (s.library.clone(), s.nid.clone()))
            .collect()
    }

    /// Maps the trap region that holds every stub handed out so far.
    pub fn map(&self, mem: &mut Memory) -> Result<(), MapError> {
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
            name: None,
            library: stub.library.clone(),
        })?;
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

impl Resolver for Hle {
    fn resolve(&mut self, import: &Import<'_>) -> Option<u64> {
        Some(self.stub(import.nid, &import.library.name))
    }
}

#[cfg(test)]
mod tests {
    use memory::Prot;

    use super::*;

    fn guest() -> (Cpu, Memory) {
        let mut mem = Memory::new(1 << 40).unwrap();
        mem.map(0x10_0000, PAGE_SIZE, Prot::RW, "stack").unwrap();
        mem.map(0x20_0000, PAGE_SIZE, Prot::RW, "data").unwrap();
        let mut cpu = Cpu::default();
        cpu.regs[RSP] = 0x10_0000 + PAGE_SIZE - 8;
        mem.write_u64(cpu.regs[RSP], 0x40_1234).unwrap();
        (cpu, mem)
    }

    fn call(
        hle: &Hle,
        stub: u64,
        cpu: &mut Cpu,
        mem: &mut Memory,
        out: &mut Vec<u8>,
    ) -> Result<Outcome, CallError> {
        let mut err = Vec::new();
        let mut env = Env {
            cpu,
            mem,
            stdout: out,
            stderr: &mut err,
        };
        hle.call(stub, &mut env)
    }

    #[test]
    fn write_copies_guest_bytes_to_stdout_and_returns() {
        let mut hle = Hle::new();
        let stub = hle.stub_by_name("libkernel", "write");
        let (mut cpu, mut mem) = guest();
        mem.write(0x20_0000, b"hi\n").unwrap();
        cpu.regs[RDI] = 1;
        cpu.regs[RSI] = 0x20_0000;
        cpu.regs[RDX] = 3;
        let mut out = Vec::new();
        assert_eq!(
            call(&hle, stub, &mut cpu, &mut mem, &mut out).unwrap(),
            Outcome::Return(3)
        );
        assert_eq!(out, b"hi\n");
        assert_eq!(
            (cpu.rip, cpu.regs[RAX], cpu.regs[RSP]),
            (0x40_1234, 3, 0x10_0000 + PAGE_SIZE)
        );
    }

    #[test]
    fn exit_reports_the_status() {
        let mut hle = Hle::new();
        let stub = hle.stub_by_name("libkernel", "exit");
        let (mut cpu, mut mem) = guest();
        cpu.regs[RDI] = 7;
        assert_eq!(
            call(&hle, stub, &mut cpu, &mut mem, &mut Vec::new()).unwrap(),
            Outcome::Exit(7)
        );
    }

    #[test]
    fn unknown_functions_get_a_stub_that_reports_them() {
        let mut hle = Hle::new();
        let stub = hle.stub("AAAAAAAAAAA", "libSceFoo");
        assert_eq!(
            hle.unimplemented(),
            [("libSceFoo".to_owned(), "AAAAAAAAAAA".to_owned())]
        );
        let (mut cpu, mut mem) = guest();
        assert!(matches!(
            call(&hle, stub, &mut cpu, &mut mem, &mut Vec::new()),
            Err(CallError::Unimplemented { .. })
        ));
    }

    #[test]
    fn variables_bind_to_readable_data() {
        let mut hle = Hle::new();
        let guard = hle.stub_by_name("libkernel", "__stack_chk_guard");
        assert_eq!(hle.stub_by_name("libkernel", "__stack_chk_guard"), guard);
        let mut mem = Memory::new(1 << 40).unwrap();
        hle.map(&mut mem).unwrap();
        assert_ne!(mem.read_u64(guard).unwrap(), 0);
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
        let (mut cpu, mut mem) = guest();
        cpu.regs[RDI] = 1;
        cpu.regs[RSI] = 0x9999_0000;
        cpu.regs[RDX] = 4;
        assert!(matches!(
            call(&hle, stub, &mut cpu, &mut mem, &mut Vec::new()),
            Err(CallError::Memory(_))
        ));
    }
}
