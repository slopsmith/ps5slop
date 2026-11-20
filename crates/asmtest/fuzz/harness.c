// Runs instruction tests natively for ps5slop-fuzz: under Rosetta 2 on an
// Apple Silicon Mac, or on an x86-64 machine. Build with
// `clang -arch x86_64 -O1 harness.c`. Input and output are streams of cases:
//   u8 len, u8 code[15], u32 states, then `states` State records.
// The output repeats each State after the code ran, with `status` set.

#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

typedef struct {
    uint64_t r[16];
    uint64_t flags;
    uint64_t status;
    uint8_t y[16][32];
} State;

State *g_state;
void *g_code;
uint64_t g_saved_rsp;
void run(void);

__asm__(
    ".globl _run\n_run:\n"
    "push %rbx\npush %rbp\npush %r12\npush %r13\npush %r14\npush %r15\n"
    "mov %rsp, _g_saved_rsp(%rip)\n"
    "mov _g_state(%rip), %rax\n"
    "vmovdqu 0x90(%rax), %ymm0\nvmovdqu 0xb0(%rax), %ymm1\nvmovdqu 0xd0(%rax), %ymm2\nvmovdqu 0xf0(%rax), %ymm3\n"
    "vmovdqu 0x110(%rax), %ymm4\nvmovdqu 0x130(%rax), %ymm5\nvmovdqu 0x150(%rax), %ymm6\nvmovdqu 0x170(%rax), %ymm7\n"
    "vmovdqu 0x190(%rax), %ymm8\nvmovdqu 0x1b0(%rax), %ymm9\nvmovdqu 0x1d0(%rax), %ymm10\nvmovdqu 0x1f0(%rax), %ymm11\n"
    "vmovdqu 0x210(%rax), %ymm12\nvmovdqu 0x230(%rax), %ymm13\nvmovdqu 0x250(%rax), %ymm14\nvmovdqu 0x270(%rax), %ymm15\n"
    "pushq 0x80(%rax)\npopfq\n"
    "mov 0x08(%rax), %rcx\nmov 0x10(%rax), %rdx\nmov 0x18(%rax), %rbx\nmov 0x28(%rax), %rbp\n"
    "mov 0x30(%rax), %rsi\nmov 0x38(%rax), %rdi\nmov 0x40(%rax), %r8\nmov 0x48(%rax), %r9\n"
    "mov 0x50(%rax), %r10\nmov 0x58(%rax), %r11\nmov 0x60(%rax), %r12\nmov 0x68(%rax), %r13\n"
    "mov 0x70(%rax), %r14\nmov 0x78(%rax), %r15\nmov 0x00(%rax), %rax\n"
    "call *_g_code(%rip)\n"
    "push %rax\n"
    "mov _g_state(%rip), %rax\n"
    "mov %rcx, 0x08(%rax)\nmov %rdx, 0x10(%rax)\nmov %rbx, 0x18(%rax)\nmov %rbp, 0x28(%rax)\n"
    "mov %rsi, 0x30(%rax)\nmov %rdi, 0x38(%rax)\nmov %r8, 0x40(%rax)\nmov %r9, 0x48(%rax)\n"
    "mov %r10, 0x50(%rax)\nmov %r11, 0x58(%rax)\nmov %r12, 0x60(%rax)\nmov %r13, 0x68(%rax)\n"
    "mov %r14, 0x70(%rax)\nmov %r15, 0x78(%rax)\n"
    "pushfq\npopq 0x80(%rax)\ncld\n"
    "popq 0x00(%rax)\n"
    "vmovdqu %ymm0, 0x90(%rax)\nvmovdqu %ymm1, 0xb0(%rax)\nvmovdqu %ymm2, 0xd0(%rax)\nvmovdqu %ymm3, 0xf0(%rax)\n"
    "vmovdqu %ymm4, 0x110(%rax)\nvmovdqu %ymm5, 0x130(%rax)\nvmovdqu %ymm6, 0x150(%rax)\nvmovdqu %ymm7, 0x170(%rax)\n"
    "vmovdqu %ymm8, 0x190(%rax)\nvmovdqu %ymm9, 0x1b0(%rax)\nvmovdqu %ymm10, 0x1d0(%rax)\nvmovdqu %ymm11, 0x1f0(%rax)\n"
    "vmovdqu %ymm12, 0x210(%rax)\nvmovdqu %ymm13, 0x230(%rax)\nvmovdqu %ymm14, 0x250(%rax)\nvmovdqu %ymm15, 0x270(%rax)\n"
    "mov _g_saved_rsp(%rip), %rsp\n"
    "pop %r15\npop %r14\npop %r13\npop %r12\npop %rbp\npop %rbx\n"
    "vzeroupper\nret\n");

static sigjmp_buf g_jump;

static void on_signal(int sig) { siglongjmp(g_jump, sig); }

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: harness IN OUT\n");
        return 2;
    }
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
    if (!in || !out) {
        perror("harness");
        return 2;
    }
    struct sigaction sa = {0};
    sa.sa_handler = on_signal;
    sa.sa_flags = SA_NODEFER;
    // Not SIGBUS or SIGSEGV: Rosetta uses them to notice code being rewritten.
    sigaction(SIGILL, &sa, 0);
    sigaction(SIGFPE, &sa, 0);
    sigaction(SIGTRAP, &sa, 0);
    static State s;
    for (;;) {
        uint8_t head[16];
        uint32_t states;
        if (fread(head, 1, 16, in) != 16 || fread(&states, 4, 1, in) != 1)
            break;
        // A fresh page per case: Rosetta faults on writes to code it translated.
        uint8_t *page = mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (page == MAP_FAILED) {
            perror("mmap");
            return 2;
        }
        memcpy(page, head + 1, head[0]);
        page[head[0]] = 0xc3;
        fwrite(head, 1, 16, out);
        fwrite(&states, 4, 1, out);
        for (uint32_t i = 0; i < states; i++) {
            if (fread(&s, sizeof s, 1, in) != 1)
                return 2;
            g_state = &s;
            g_code = page;
            int sig = sigsetjmp(g_jump, 1);
            if (sig == 0) {
                run();
                s.status = 0;
            } else {
                __asm__ volatile("cld");
                s.status = (uint64_t)sig;
            }
            fwrite(&s, sizeof s, 1, out);
        }
    }
    fclose(out);
    return 0;
}
