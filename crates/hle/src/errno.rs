//! errno values. Orbis is FreeBSD-derived; these are FreeBSD's numbers
//! (`sys/errno.h`), assumed rather than verified for Orbis.

pub const EPERM: i32 = 1;
pub const ESRCH: i32 = 3;
pub const ENOENT: i32 = 2;
pub const EIO: i32 = 5;
pub const EBADF: i32 = 9;
pub const EDEADLK: i32 = 11;
pub const ENOMEM: i32 = 12;
pub const EACCES: i32 = 13;
pub const EBUSY: i32 = 16;
pub const EEXIST: i32 = 17;
pub const ENOTDIR: i32 = 20;
pub const EISDIR: i32 = 21;
pub const EINVAL: i32 = 22;
pub const ENOTTY: i32 = 25;
pub const ENOSPC: i32 = 28;
pub const EROFS: i32 = 30;
pub const ERANGE: i32 = 34;
pub const EAGAIN: i32 = 35;
pub const ETIMEDOUT: i32 = 60;
pub const ENOSYS: i32 = 78;
pub const EAFNOSUPPORT: i32 = 47;
pub const ENETDOWN: i32 = 50;
pub const EOVERFLOW: i32 = 84;
pub const ECANCELED: i32 = 85;
pub const EILSEQ: i32 = 86;
