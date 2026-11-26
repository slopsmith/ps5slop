//! The few libdrm_amdgpu calls a raw compute submission needs (MIT-licensed
//! library, declared here from its public `amdgpu.h`).

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr::null_mut;

type Handle = *mut c_void;

#[repr(C)]
struct AllocRequest {
    alloc_size: u64,
    phys_alignment: u64,
    preferred_heap: u32,
    flags: u64,
}

#[repr(C)]
struct Fence {
    context: Handle,
    ip_type: u32,
    ip_instance: u32,
    ring: u32,
    fence: u64,
}

#[repr(C)]
struct IbInfo {
    flags: u64,
    ib_mc_address: u64,
    size: u32,
}

#[repr(C)]
struct FenceInfo {
    handle: Handle,
    offset: u64,
}

#[repr(C)]
struct CsRequest {
    flags: u64,
    ip_type: u32,
    ip_instance: u32,
    ring: u32,
    resources: Handle,
    number_of_dependencies: u32,
    dependencies: *mut Fence,
    number_of_ibs: u32,
    ibs: *mut IbInfo,
    seq_no: u64,
    fence_info: FenceInfo,
}

#[link(name = "drm_amdgpu")]
unsafe extern "C" {
    fn amdgpu_device_initialize(
        fd: c_int,
        major: *mut u32,
        minor: *mut u32,
        dev: *mut Handle,
    ) -> c_int;
    fn amdgpu_device_deinitialize(dev: Handle) -> c_int;
    fn amdgpu_get_marketing_name(dev: Handle) -> *const c_char;
    fn amdgpu_bo_alloc(dev: Handle, req: *const AllocRequest, bo: *mut Handle) -> c_int;
    fn amdgpu_bo_free(bo: Handle) -> c_int;
    fn amdgpu_bo_cpu_map(bo: Handle, cpu: *mut *mut c_void) -> c_int;
    fn amdgpu_bo_cpu_unmap(bo: Handle) -> c_int;
    fn amdgpu_bo_va_op_raw(
        dev: Handle,
        bo: Handle,
        offset: u64,
        size: u64,
        addr: u64,
        flags: u64,
        ops: u32,
    ) -> c_int;
    fn amdgpu_cs_ctx_create(dev: Handle, ctx: *mut Handle) -> c_int;
    fn amdgpu_cs_ctx_free(ctx: Handle) -> c_int;
    fn amdgpu_bo_list_create(
        dev: Handle,
        count: u32,
        bos: *const Handle,
        priorities: *const u8,
        list: *mut Handle,
    ) -> c_int;
    fn amdgpu_bo_list_destroy(list: Handle) -> c_int;
    fn amdgpu_cs_submit(ctx: Handle, flags: u64, req: *mut CsRequest, count: u32) -> c_int;
    fn amdgpu_cs_query_fence_status(
        fence: *const Fence,
        timeout_ns: u64,
        flags: u64,
        expired: *mut u32,
    ) -> c_int;
}

const GEM_DOMAIN_GTT: u32 = 2;
const GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1;
const VA_OP_MAP: u32 = 1;
const VA_OP_UNMAP: u32 = 2;
const VM_PAGE_READABLE: u64 = 1 << 1;
const VM_PAGE_WRITEABLE: u64 = 1 << 2;
const VM_PAGE_EXECUTABLE: u64 = 1 << 3;
const VM_PAGE_PRT: u64 = 1 << 4;
/// The compute queue.
pub const HW_IP_COMPUTE: u32 = 1;
/// The graphics queue.
pub const HW_IP_GFX: u32 = 0;

fn check(what: &str, r: c_int) -> Result<(), String> {
    if r == 0 {
        Ok(())
    } else {
        Err(format!(
            "{what} failed: {}",
            std::io::Error::from_raw_os_error(-r)
        ))
    }
}

/// An amdgpu device opened through its render node.
pub struct Device {
    handle: Handle,
    _file: File,
}

impl Device {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::options()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let (mut major, mut minor, mut handle) = (0, 0, null_mut());
        // SAFETY: the out pointers are valid; libdrm duplicates the descriptor.
        check("amdgpu_device_initialize", unsafe {
            amdgpu_device_initialize(file.as_raw_fd(), &mut major, &mut minor, &mut handle)
        })?;
        Ok(Self {
            handle,
            _file: file,
        })
    }

    pub fn name(&self) -> String {
        // SAFETY: libdrm returns a static string or null.
        let p = unsafe { amdgpu_get_marketing_name(self.handle) };
        if p.is_null() {
            return "unknown AMD GPU".into();
        }
        // SAFETY: non-null, NUL-terminated, static.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// CPU-visible memory mapped at GPU address `addr`.
    pub fn buffer(&self, addr: u64, size: u64) -> Result<Buffer<'_>, String> {
        let req = AllocRequest {
            alloc_size: size,
            phys_alignment: 4096,
            preferred_heap: GEM_DOMAIN_GTT,
            flags: GEM_CREATE_CPU_ACCESS_REQUIRED,
        };
        let mut bo = null_mut();
        // SAFETY: valid device and out pointer.
        check("amdgpu_bo_alloc", unsafe {
            amdgpu_bo_alloc(self.handle, &req, &mut bo)
        })?;
        let mut buffer = Buffer {
            device: self,
            bo,
            addr,
            size,
            cpu: null_mut(),
            mapped: false,
        };
        let flags = VM_PAGE_READABLE | VM_PAGE_WRITEABLE | VM_PAGE_EXECUTABLE;
        // SAFETY: `bo` is a live buffer of `size` bytes.
        check(&format!("mapping {addr:#x}+{size:#x}"), unsafe {
            amdgpu_bo_va_op_raw(self.handle, bo, 0, size, addr, flags, VA_OP_MAP)
        })?;
        buffer.mapped = true;
        let mut cpu = null_mut();
        // SAFETY: `bo` is live; the mapping lives until `cpu_unmap` in drop.
        check("amdgpu_bo_cpu_map", unsafe {
            amdgpu_bo_cpu_map(bo, &mut cpu)
        })?;
        buffer.cpu = cpu.cast();
        Ok(buffer)
    }

    /// Maps `[addr, addr + size)` with no memory: reads give zeros and writes vanish.
    pub fn unbacked(&self, addr: u64, size: u64) -> Result<(), String> {
        // SAFETY: a PRT mapping needs no buffer.
        check(&format!("unbacked mapping {addr:#x}+{size:#x}"), unsafe {
            amdgpu_bo_va_op_raw(
                self.handle,
                null_mut(),
                0,
                size,
                addr,
                VM_PAGE_PRT,
                VA_OP_MAP,
            )
        })
    }

    /// Runs the command buffer in `ib` (its first `dwords` dwords) on queue
    /// `ip` with `buffers` resident, waiting up to `timeout`. False on a timeout.
    pub fn run(
        &self,
        ip: u32,
        ib: &Buffer,
        dwords: u32,
        buffers: &[&Buffer],
        timeout: std::time::Duration,
    ) -> Result<bool, String> {
        let mut ctx = null_mut();
        // SAFETY: valid device and out pointer.
        check("amdgpu_cs_ctx_create", unsafe {
            amdgpu_cs_ctx_create(self.handle, &mut ctx)
        })?;
        let handles: Vec<Handle> = buffers.iter().chain([&ib]).map(|b| b.bo).collect();
        let mut list = null_mut();
        // SAFETY: `handles` are live buffers of this device.
        let created = check("amdgpu_bo_list_create", unsafe {
            amdgpu_bo_list_create(
                self.handle,
                handles.len() as u32,
                handles.as_ptr(),
                std::ptr::null(),
                &mut list,
            )
        });
        let result = created.and_then(|()| {
            let mut info = IbInfo {
                flags: 0,
                ib_mc_address: ib.addr,
                size: dwords,
            };
            let mut req = CsRequest {
                flags: 0,
                ip_type: ip,
                ip_instance: 0,
                ring: 0,
                resources: list,
                number_of_dependencies: 0,
                dependencies: null_mut(),
                number_of_ibs: 1,
                ibs: &mut info,
                seq_no: 0,
                fence_info: FenceInfo {
                    handle: null_mut(),
                    offset: 0,
                },
            };
            // SAFETY: the request points at live data for the call.
            check("amdgpu_cs_submit", unsafe {
                amdgpu_cs_submit(ctx, 0, &mut req, 1)
            })?;
            let fence = Fence {
                context: ctx,
                ip_type: ip,
                ip_instance: 0,
                ring: 0,
                fence: req.seq_no,
            };
            let mut expired = 0;
            const ETIME: c_int = 62;
            // SAFETY: `fence` names the submission above.
            let r = unsafe {
                amdgpu_cs_query_fence_status(&fence, timeout.as_nanos() as u64, 0, &mut expired)
            };
            if r == -ETIME {
                return Ok(false);
            }
            check("amdgpu_cs_query_fence_status", r)?;
            Ok(expired != 0)
        });
        // SAFETY: created above; destroyed once.
        unsafe {
            if !list.is_null() {
                amdgpu_bo_list_destroy(list);
            }
            amdgpu_cs_ctx_free(ctx);
        }
        result
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: initialized in `open`, released once.
        unsafe { amdgpu_device_deinitialize(self.handle) };
    }
}

/// A GPU buffer at a fixed GPU address, also mapped for the CPU.
pub struct Buffer<'a> {
    device: &'a Device,
    bo: Handle,
    pub addr: u64,
    pub size: u64,
    cpu: *mut u8,
    mapped: bool,
}

impl Buffer<'_> {
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `cpu` maps `size` bytes for the buffer's lifetime.
        unsafe { std::slice::from_raw_parts(self.cpu, self.size as usize) }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` makes the access unique on the CPU.
        unsafe { std::slice::from_raw_parts_mut(self.cpu, self.size as usize) }
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        // SAFETY: each step undoes one that succeeded in `Device::buffer`.
        unsafe {
            if !self.cpu.is_null() {
                amdgpu_bo_cpu_unmap(self.bo);
            }
            if self.mapped {
                amdgpu_bo_va_op_raw(
                    self.device.handle,
                    self.bo,
                    0,
                    self.size,
                    self.addr,
                    0,
                    VA_OP_UNMAP,
                );
            }
            amdgpu_bo_free(self.bo);
        }
    }
}
