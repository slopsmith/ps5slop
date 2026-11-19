// SPDX-License-Identifier: GPL-2.0-or-later
// Threads, mutexes, a condition variable and pthread_once through libkernel.

typedef void *pthread_t;
typedef void *pthread_mutex_t;
typedef void *pthread_cond_t;
typedef struct {
	int state;
	void *mutex;
} pthread_once_t;

int pthread_create(pthread_t *thread, const void *attr, void *(*start)(void *), void *arg);
int pthread_join(pthread_t thread, void **result);
int pthread_mutex_lock(pthread_mutex_t *m);
int pthread_mutex_unlock(pthread_mutex_t *m);
int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m);
int pthread_cond_signal(pthread_cond_t *c);
int pthread_once(pthread_once_t *once, void (*routine)(void));
long write(int fd, const void *buf, unsigned long len);
void exit(int status) __attribute__((noreturn));

#define THREADS 4
#define ROUNDS 1000

static pthread_mutex_t lock;
static pthread_cond_t ready_cond;
static long counter;
static int ready;
static int inits;
static pthread_once_t once;

static void print(const char *s)
{
	unsigned long n = 0;
	while (s[n])
		n++;
	write(1, s, n);
}

static void print_num(long v)
{
	char buf[24];
	int i = sizeof buf;
	buf[--i] = 0;
	do {
		buf[--i] = '0' + v % 10;
		v /= 10;
	} while (v);
	print(&buf[i]);
}

static void init(void)
{
	inits++;
}

static void *worker(void *arg)
{
	pthread_once(&once, init);
	for (int i = 0; i < ROUNDS; i++) {
		pthread_mutex_lock(&lock);
		counter++;
		pthread_mutex_unlock(&lock);
	}
	return (void *)((long)arg * 2);
}

static void *signaller(void *arg)
{
	pthread_mutex_lock(&lock);
	ready = 1;
	pthread_cond_signal(&ready_cond);
	pthread_mutex_unlock(&lock);
	return arg;
}

void _start(void)
{
	pthread_t threads[THREADS];
	long sum = 0;
	for (long i = 0; i < THREADS; i++)
		pthread_create(&threads[i], 0, worker, (void *)i);
	for (int i = 0; i < THREADS; i++) {
		void *result;
		pthread_join(threads[i], &result);
		sum += (long)result;
	}

	pthread_t s;
	pthread_mutex_lock(&lock);
	pthread_create(&s, 0, signaller, 0);
	while (!ready)
		pthread_cond_wait(&ready_cond, &lock);
	pthread_mutex_unlock(&lock);
	pthread_join(s, 0);

	print("counter ");
	print_num(counter);
	print(", inits ");
	print_num(inits);
	print(", results ");
	print_num(sum);
	print("\n");
	exit(counter == THREADS * ROUNDS && inits == 1 && sum == 12 ? 0 : 1);
}
