// SPDX-License-Identifier: GPL-2.0-or-later
// Prints a line and exits. Built by scripts/build-test-programs.

long write(int fd, const void *buf, unsigned long len);
void exit(int status) __attribute__((noreturn));

void _start(void)
{
	static const char msg[] = "hello from ps5slop\n";
	write(1, msg, sizeof msg - 1);
	exit(0);
}
