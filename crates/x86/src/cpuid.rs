//! What CPUID reports: a Zen 2 core (family 17h), as in the PS5. Feature bit
//! positions come from AMD's public CPUID specification and APM volume 3.
//! The PS5's own CPUID values are unknown; this is a plausible Zen 2, not a
//! copy of one.

/// `(eax, ebx, ecx, edx)` for `leaf` and `subleaf`.
pub fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    let bits = |list: &[u32]| list.iter().fold(0u32, |acc, b| acc | 1 << b);
    match (leaf, subleaf) {
        (0, _) => (
            0xd,
            u32::from_le_bytes(*b"Auth"),
            u32::from_le_bytes(*b"cAMD"),
            u32::from_le_bytes(*b"enti"),
        ),
        (1, _) => {
            // family 17h (0xf + 0x8), model 0x71, stepping 0
            let eax = 0x0087_0f10;
            let ebx = 8 << 16 | 8 << 8;
            // SSE3 PCLMULQDQ SSSE3 FMA CX16 SSE4.1 SSE4.2 MOVBE POPCNT AES XSAVE OSXSAVE AVX F16C RDRAND
            let ecx = bits(&[0, 1, 9, 12, 13, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30]);
            // FPU TSC CX8 CMOV CLFLUSH MMX FXSR SSE SSE2
            let edx = bits(&[0, 4, 8, 15, 19, 23, 24, 25, 26]);
            (eax, ebx, ecx, edx)
        }
        (7, 0) => {
            // FSGSBASE BMI1 AVX2 BMI2 RDSEED ADX CLFLUSHOPT CLWB SHA
            let ebx = bits(&[0, 3, 5, 8, 18, 19, 23, 24, 29]);
            // RDPID
            let ecx = bits(&[22]);
            (0, ebx, ecx, 0)
        }
        (0xd, 0) => (0x7, 0x340, 0x340, 0),
        (0xd, 1) => (bits(&[0, 1, 3]), 0, 0, 0),
        (0x8000_0000, _) => (
            0x8000_0008,
            u32::from_le_bytes(*b"Auth"),
            u32::from_le_bytes(*b"cAMD"),
            u32::from_le_bytes(*b"enti"),
        ),
        (0x8000_0001, _) => {
            // LAHF/SAHF ABM(LZCNT) SSE4A MisalignSSE PREFETCHW
            let ecx = bits(&[0, 5, 6, 7, 8]);
            // SYSCALL NX MMXEXT FFXSR RDTSCP LM
            let edx = bits(&[11, 20, 22, 25, 27, 29]);
            (0x0087_0f10, 0, ecx, edx)
        }
        (0x8000_0008, _) => (0x3030, bits(&[0]), 0, 0),
        _ => (0, 0, 0, 0),
    }
}
