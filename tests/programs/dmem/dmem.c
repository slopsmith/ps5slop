// SPDX-License-Identifier: GPL-2.0-or-later
// Direct memory: allocate, map, find the offset again through
// sceKernelVirtualQuery and release it there, as Minecraft's allocator does.

typedef struct {
	unsigned long start;
	unsigned long end;
	unsigned long offset;
	int protection;
	int memory_type;
	unsigned char flags;
	char name[32];
} query_info;

int sceKernelAllocateMainDirectMemory(unsigned long len, unsigned long align, int type, long *start);
int sceKernelMapDirectMemory(void **addr, unsigned long len, int prot, int flags, long start, unsigned long align);
int sceKernelReleaseDirectMemory(long start, unsigned long len);
int sceKernelVirtualQuery(const void *addr, int flags, query_info *info, unsigned long size);
int sceKernelDirectMemoryQuery(long offset, int flags, void *info, unsigned long size);
int sceKernelReserveVirtualRange(void **addr, unsigned long len, int flags, unsigned long align);
long write(int fd, const void *buf, unsigned long len);
void exit(int status) __attribute__((noreturn));

#define BLOCK 0x10000
#define BLOCKS 3
#define TYPE 3
#define MAP_FIXED 0x10

static void print(const char *s)
{
	unsigned long n = 0;
	while (s[n])
		n++;
	write(1, s, n);
}

static void check(int ok, const char *what)
{
	print(ok ? "ok " : "FAIL ");
	print(what);
	print("\n");
	if (!ok)
		exit(1);
}

int main(void)
{
	long offsets[BLOCKS];
	void *addrs[BLOCKS];
	query_info info;
	for (int i = 0; i < BLOCKS; i++) {
		addrs[i] = 0;
		if (sceKernelAllocateMainDirectMemory(BLOCK, BLOCK, TYPE, &offsets[i]) != 0 ||
		    sceKernelMapDirectMemory(&addrs[i], BLOCK, 3, 0, offsets[i], 0) != 0)
			check(0, "allocate and map");
		((volatile char *)addrs[i])[0] = 1;
	}
	check(offsets[0] != offsets[1] && offsets[1] != offsets[2], "allocations differ");

	int found = 1;
	for (int i = 0; i < BLOCKS; i++) {
		if (sceKernelVirtualQuery(addrs[i], 0, &info, sizeof info) != 0 ||
		    info.start != (unsigned long)addrs[i] || info.offset != (unsigned long)offsets[i] ||
		    info.memory_type != TYPE || !(info.flags & 2))
			found = 0;
	}
	check(found, "virtual query reports offset and type");

	if (sceKernelVirtualQuery((char *)addrs[1] + 0x4000, 0, &info, sizeof info) != 0)
		check(0, "query inside a mapping");

	void *at = addrs[1];
	if (sceKernelReserveVirtualRange(&at, BLOCK, MAP_FIXED, 0) != 0 ||
	    sceKernelReleaseDirectMemory(info.offset, BLOCK) != 0)
		check(0, "unmap and release");
	unsigned long range[3];
	check(sceKernelDirectMemoryQuery(offsets[1], 0, range, sizeof range) != 0 &&
	      sceKernelDirectMemoryQuery(offsets[0], 0, range, sizeof range) == 0 &&
	      sceKernelDirectMemoryQuery(offsets[2], 0, range, sizeof range) == 0,
	      "only the queried block was released");
	return 0;
}

void _start(void)
{
	exit(main());
}
