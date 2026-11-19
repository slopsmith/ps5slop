// Prints the page size and which guest-relevant addresses a native arm64
// process can map with MAP_FIXED. See docs/memory.md.
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
	printf("page size: %ld\n", sysconf(_SC_PAGESIZE));

	void *p = mmap(NULL, 0x4000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	printf("mprotect 4 KiB: %s\n", mprotect((char *)p + 0x1000, 0x1000, PROT_READ) == 0 ? "ok" : "rejected");

	const unsigned long addrs[] = {
		0x400000UL, 0x100000000UL, 0x800000000UL, 0x1000000000UL,
		0x10000000000UL, 0x7f0000000000UL,
	};
	for (unsigned i = 0; i < sizeof addrs / sizeof addrs[0]; i++) {
		void *m = mmap((void *)addrs[i], 1 << 20, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		printf("MAP_FIXED %#14lx: %s\n", addrs[i], m == MAP_FAILED ? "failed" : "ok");
	}

	fflush(stdout);
	char cmd[64];
	snprintf(cmd, sizeof cmd, "vmmap -interleaved %d", getpid());
	return system(cmd) == 0 ? 0 : 1;
}
