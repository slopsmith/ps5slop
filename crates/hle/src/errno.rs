//! errno values. Orbis is FreeBSD-derived; these are FreeBSD's numbers
//! (`sys/errno.h`), assumed rather than verified for Orbis.

pub const EPERM: i32 = 1;
pub const ENOENT: i32 = 2;
pub const EBADF: i32 = 9;
pub const EDEADLK: i32 = 11;
pub const ENOMEM: i32 = 12;
pub const EBUSY: i32 = 16;
pub const EINVAL: i32 = 22;
pub const ENOTTY: i32 = 25;
pub const EAGAIN: i32 = 35;
pub const ETIMEDOUT: i32 = 60;
pub const ENOSYS: i32 = 78;
