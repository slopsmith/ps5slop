//! Safe wrappers around Apple's Hypervisor.framework for ARM64 guests.
//!
//! The framework allows one VM per process, and a vCPU may only be used from
//! the thread that created it; `Vcpu` is therefore neither `Send` nor `Sync`.

use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;

mod ffi {
    use std::ffi::c_void;

    pub type HvReturn = i32;
    pub type HvVcpu = u64;
    pub type HvVmConfig = *mut c_void;

    #[repr(C)]
    pub struct ExitException {
        pub syndrome: u64,
        pub virtual_address: u64,
        pub physical_address: u64,
    }

    #[repr(C)]
    pub struct VcpuExit {
        pub reason: u32,
        pub exception: ExitException,
    }

    #[link(name = "Hypervisor", kind = "framework")]
    unsafe extern "C" {
        pub fn hv_vm_config_create() -> HvVmConfig;
        pub fn hv_vm_config_set_ipa_size(config: HvVmConfig, bits: u32) -> HvReturn;
        pub fn hv_vm_config_set_ipa_granule(config: HvVmConfig, granule: u32) -> HvReturn;
        pub fn hv_vm_config_get_max_ipa_size(bits: *mut u32) -> HvReturn;
        pub fn hv_vm_create(config: HvVmConfig) -> HvReturn;
        pub fn hv_vm_destroy() -> HvReturn;
        pub fn hv_vm_map(addr: *mut c_void, ipa: u64, size: usize, flags: u64) -> HvReturn;
        pub fn hv_vm_unmap(ipa: u64, size: usize) -> HvReturn;
        pub fn hv_vm_protect(ipa: u64, size: usize, flags: u64) -> HvReturn;
        pub fn hv_vcpu_create(
            vcpu: *mut HvVcpu,
            exit: *mut *const VcpuExit,
            config: *mut c_void,
        ) -> HvReturn;
        pub fn hv_vcpu_destroy(vcpu: HvVcpu) -> HvReturn;
        pub fn hv_vcpu_get_reg(vcpu: HvVcpu, reg: u32, value: *mut u64) -> HvReturn;
        pub fn hv_vcpu_set_reg(vcpu: HvVcpu, reg: u32, value: u64) -> HvReturn;
        pub fn hv_vcpu_get_sys_reg(vcpu: HvVcpu, reg: u16, value: *mut u64) -> HvReturn;
        pub fn hv_vcpu_set_sys_reg(vcpu: HvVcpu, reg: u16, value: u64) -> HvReturn;
        pub fn hv_vcpu_get_simd_fp_reg(vcpu: HvVcpu, reg: u32, value: *mut u128) -> HvReturn;
        pub fn hv_vcpu_set_simd_fp_reg(vcpu: HvVcpu, reg: u32, value: u128) -> HvReturn;
        pub fn hv_vcpu_run(vcpu: HvVcpu) -> HvReturn;
        pub fn hv_vcpus_exit(vcpus: *const HvVcpu, count: u32) -> HvReturn;
        pub fn hv_vcpu_set_pending_interrupt(vcpu: HvVcpu, kind: u32, pending: bool) -> HvReturn;
        pub fn hv_vcpu_set_vtimer_offset(vcpu: HvVcpu, offset: u64) -> HvReturn;
    }

    unsafe extern "C" {
        pub fn os_release(object: *mut c_void);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Error(pub i32);

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.0 as u32 {
            0xfae9_4001 => "HV_ERROR",
            0xfae9_4002 => "HV_BUSY",
            0xfae9_4003 => "HV_BAD_ARGUMENT",
            0xfae9_4005 => "HV_NO_RESOURCES",
            0xfae9_4006 => "HV_NO_DEVICE",
            0xfae9_4007 => "HV_DENIED",
            0xfae9_400f => "HV_UNSUPPORTED",
            _ => "unknown",
        };
        write!(f, "{name} ({:#x})", self.0 as u32)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)?;
        if self.0 as u32 == 0xfae9_4007 {
            f.write_str(": is the binary signed with com.apple.security.hypervisor?")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn check(r: i32) -> Result<()> {
    if r == 0 { Ok(()) } else { Err(Error(r)) }
}

pub mod mem {
    pub const READ: u64 = 1 << 0;
    pub const WRITE: u64 = 1 << 1;
    pub const EXEC: u64 = 1 << 2;
    pub const RWX: u64 = READ | WRITE | EXEC;
}

pub fn max_ipa_bits() -> Result<u32> {
    let mut bits = 0;
    check(unsafe { ffi::hv_vm_config_get_max_ipa_size(&mut bits) })?;
    Ok(bits)
}

/// The process's single virtual machine. Dropping it destroys the VM.
pub struct Vm {
    _not_send: PhantomData<*const ()>,
}

impl Vm {
    /// Creates the VM with an `ipa_bits`-wide guest physical address space
    /// and 4 KiB stage-2 pages.
    pub fn new(ipa_bits: u32) -> Result<Vm> {
        unsafe {
            let config = ffi::hv_vm_config_create();
            let result = check(ffi::hv_vm_config_set_ipa_size(config, ipa_bits))
                .and_then(|_| check(ffi::hv_vm_config_set_ipa_granule(config, 0)))
                .and_then(|_| check(ffi::hv_vm_create(config)));
            ffi::os_release(config);
            result?;
        }
        Ok(Vm {
            _not_send: PhantomData,
        })
    }

    /// Maps host memory into the guest physical address space.
    ///
    /// # Safety
    /// `host` must stay valid and page-aligned for `size` bytes until unmapped.
    pub unsafe fn map(&self, host: *mut u8, ipa: u64, size: usize, flags: u64) -> Result<()> {
        check(unsafe { ffi::hv_vm_map(host as *mut c_void, ipa, size, flags) })
    }

    pub fn unmap(&self, ipa: u64, size: usize) -> Result<()> {
        check(unsafe { ffi::hv_vm_unmap(ipa, size) })
    }

    pub fn protect(&self, ipa: u64, size: usize, flags: u64) -> Result<()> {
        check(unsafe { ffi::hv_vm_protect(ipa, size, flags) })
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        unsafe {
            ffi::hv_vm_destroy();
        }
    }
}

/// Page-aligned anonymous memory suitable for `Vm::map`.
pub struct GuestMemory {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for GuestMemory {}

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
    -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

impl GuestMemory {
    pub fn new(len: usize) -> std::io::Result<Self> {
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        const MAP_PRIVATE: i32 = 0x2;
        const MAP_ANON: i32 = 0x1000;
        let p = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if p as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(GuestMemory {
            ptr: p as *mut u8,
            len,
        })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn write_words(&mut self, offset: usize, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            self.as_mut_slice()[offset + i * 4..offset + i * 4 + 4]
                .copy_from_slice(&w.to_le_bytes());
        }
    }
}

impl Drop for GuestMemory {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

pub mod reg {
    pub const X0: u32 = 0;
    pub const PC: u32 = 31;
    pub const FPCR: u32 = 32;
    pub const FPSR: u32 = 33;
    pub const CPSR: u32 = 34;
}

/// System register IDs (`HV_SYS_REG_*`), packed as op0:op1:CRn:CRm:op2.
pub mod sysreg {
    pub const SCTLR_EL1: u16 = 0xc080;
    pub const ACTLR_EL1: u16 = 0xc081;
    pub const CPACR_EL1: u16 = 0xc082;
    pub const TTBR0_EL1: u16 = 0xc100;
    pub const TTBR1_EL1: u16 = 0xc101;
    pub const TCR_EL1: u16 = 0xc102;
    pub const SPSR_EL1: u16 = 0xc200;
    pub const ELR_EL1: u16 = 0xc201;
    pub const SP_EL0: u16 = 0xc208;
    pub const ESR_EL1: u16 = 0xc290;
    pub const FAR_EL1: u16 = 0xc300;
    pub const MAIR_EL1: u16 = 0xc510;
    pub const VBAR_EL1: u16 = 0xc600;
    pub const TPIDR_EL1: u16 = 0xc684;
    pub const SP_EL1: u16 = 0xe208;
    pub const TPIDR_EL0: u16 = 0xde82;
    pub const ID_AA64MMFR0_EL1: u16 = 0xc038;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// `hv_vcpus_exit` was called on this vCPU.
    Canceled,
    /// A synchronous exception to EL2 (HVC, stage-2 fault, trapped access...).
    Exception {
        syndrome: u64,
        va: u64,
        ipa: u64,
    },
    VtimerActivated,
    Unknown,
}

impl Exit {
    /// Exception class (ESR bits 26..=31) for `Exception` exits.
    pub fn ec(&self) -> Option<u32> {
        match self {
            Exit::Exception { syndrome, .. } => Some((syndrome >> 26) as u32 & 0x3f),
            _ => None,
        }
    }
}

/// A handle that may be sent to other threads to force a vCPU out of `run`.
#[derive(Clone, Copy, Debug)]
pub struct VcpuKicker(ffi::HvVcpu);

impl VcpuKicker {
    pub fn kick(&self) {
        unsafe {
            ffi::hv_vcpus_exit(&self.0, 1);
        }
    }
}

pub struct Vcpu {
    id: ffi::HvVcpu,
    exit: *const ffi::VcpuExit,
}

impl Vcpu {
    pub fn new(_vm: &Vm) -> Result<Vcpu> {
        let mut id = 0;
        let mut exit = std::ptr::null();
        check(unsafe { ffi::hv_vcpu_create(&mut id, &mut exit, std::ptr::null_mut()) })?;
        Ok(Vcpu { id, exit })
    }

    pub fn kicker(&self) -> VcpuKicker {
        VcpuKicker(self.id)
    }

    pub fn reg(&self, r: u32) -> Result<u64> {
        let mut v = 0;
        check(unsafe { ffi::hv_vcpu_get_reg(self.id, r, &mut v) })?;
        Ok(v)
    }

    pub fn set_reg(&self, r: u32, v: u64) -> Result<()> {
        check(unsafe { ffi::hv_vcpu_set_reg(self.id, r, v) })
    }

    pub fn sys_reg(&self, r: u16) -> Result<u64> {
        let mut v = 0;
        check(unsafe { ffi::hv_vcpu_get_sys_reg(self.id, r, &mut v) })?;
        Ok(v)
    }

    pub fn set_sys_reg(&self, r: u16, v: u64) -> Result<()> {
        check(unsafe { ffi::hv_vcpu_set_sys_reg(self.id, r, v) })
    }

    pub fn simd_reg(&self, r: u32) -> Result<u128> {
        let mut v = 0;
        check(unsafe { ffi::hv_vcpu_get_simd_fp_reg(self.id, r, &mut v) })?;
        Ok(v)
    }

    pub fn set_simd_reg(&self, r: u32, v: u128) -> Result<()> {
        check(unsafe { ffi::hv_vcpu_set_simd_fp_reg(self.id, r, v) })
    }

    /// Sets or clears a pending IRQ (`fiq` false) or FIQ for the next `run`.
    pub fn set_pending_interrupt(&self, fiq: bool, pending: bool) -> Result<()> {
        check(unsafe { ffi::hv_vcpu_set_pending_interrupt(self.id, fiq as u32, pending) })
    }

    /// Makes the guest's CNTVCT_EL0 read `mach_absolute_time() - offset`.
    pub fn set_vtimer_offset(&self, offset: u64) -> Result<()> {
        check(unsafe { ffi::hv_vcpu_set_vtimer_offset(self.id, offset) })
    }

    /// Selects x86-TSO memory ordering (ACTLR_EL1.EnTSO, macOS 15+).
    pub fn set_tso(&self, on: bool) -> Result<()> {
        self.set_sys_reg(sysreg::ACTLR_EL1, if on { 2 } else { 0 })
    }

    pub fn run(&mut self) -> Result<Exit> {
        check(unsafe { ffi::hv_vcpu_run(self.id) })?;
        let e = unsafe { &*self.exit };
        Ok(match e.reason {
            0 => Exit::Canceled,
            1 => Exit::Exception {
                syndrome: e.exception.syndrome,
                va: e.exception.virtual_address,
                ipa: e.exception.physical_address,
            },
            2 => Exit::VtimerActivated,
            _ => Exit::Unknown,
        })
    }
}

impl Drop for Vcpu {
    fn drop(&mut self) {
        unsafe {
            ffi::hv_vcpu_destroy(self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use a64::{Asm, Size, r};

    #[test]
    fn runs_guest_code_and_exits_on_hvc() {
        let vm = Vm::new(36).unwrap();
        let mut mem = GuestMemory::new(0x10000).unwrap();
        let mut a = Asm::new();
        a.add_imm(Size::X, r(0), r(0), 41);
        a.add_imm(Size::X, r(0), r(0), 1);
        a.hvc(0x55);
        mem.write_words(0, &a.finish());
        unsafe {
            vm.map(mem.as_ptr(), 0x4000_0000, mem.len(), mem::RWX)
                .unwrap()
        };
        let mut cpu = Vcpu::new(&vm).unwrap();
        cpu.set_reg(reg::CPSR, 0x3c5).unwrap();
        cpu.set_reg(reg::PC, 0x4000_0000).unwrap();
        cpu.set_reg(reg::X0, 0).unwrap();
        cpu.set_tso(true).unwrap();
        let exit = cpu.run().unwrap();
        assert_eq!(exit.ec(), Some(0x16));
        if let Exit::Exception { syndrome, .. } = exit {
            assert_eq!(syndrome & 0xffff, 0x55);
        }
        assert_eq!(cpu.reg(reg::X0).unwrap(), 42);
        assert_eq!(cpu.reg(reg::PC).unwrap(), 0x4000_000c);
    }
}
