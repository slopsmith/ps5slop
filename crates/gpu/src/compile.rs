//! Work that runs on a pool of compile threads, so the command processor
//! does not stall while Metal compiles programs and pipelines.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

type Job<T> = Box<dyn FnOnce() -> Option<T> + Send>;

static BLOCKED: AtomicUsize = AtomicUsize::new(0);

/// Whether some command processor is waiting for a compile. A queue waiting
/// for another queue's label does not count that time against its limit.
pub fn blocked() -> bool {
    BLOCKED.load(Ordering::Relaxed) != 0
}

enum Status<T> {
    Queued(Job<T>),
    Running,
    Done(Option<T>),
}

/// A result being computed on the pool; `None` inside means the work failed.
pub struct Task<T> {
    status: Mutex<Status<T>>,
    done: Condvar,
    /// Run once the task is done.
    then: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
}

/// Something a task can wait for before it is queued.
pub trait Dependency: Send + Sync {
    /// Runs `f` when done, at once if done already.
    fn on_done(&self, f: Box<dyn FnOnce() + Send>);
}

trait Run: Send + Sync {
    fn run(&self);
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl<T: Clone + Send + 'static> Task<T> {
    /// Queues `job` on the pool.
    pub fn spawn(job: impl FnOnce() -> Option<T> + Send + 'static) -> Arc<Self> {
        let task = Self::new(job);
        pool().push(task.clone());
        task
    }

    /// Queues `job` once every task in `after` is done, ahead of other work,
    /// so it never holds a compile thread while it waits for them.
    pub fn spawn_after(
        after: &[&dyn Dependency],
        job: impl FnOnce() -> Option<T> + Send + 'static,
    ) -> Arc<Self> {
        let task = Self::new(job);
        let left = Arc::new(std::sync::atomic::AtomicUsize::new(after.len() + 1));
        let release = {
            let (task, left) = (task.clone(), left.clone());
            move || {
                if left.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                    pool().push_front(task);
                }
            }
        };
        for dep in after {
            let release = release.clone();
            dep.on_done(Box::new(release));
        }
        release();
        task
    }

    /// Moves the task to the front of the queue if no thread has started it:
    /// work asked for again goes before work asked for once (such as the
    /// shader warm-up draws titles make at start-up).
    pub fn hurry(self: &Arc<Self>) {
        if matches!(*lock(&self.status), Status::Queued(_)) {
            pool().move_to_front(Arc::as_ptr(self).cast());
        }
    }

    fn new(job: impl FnOnce() -> Option<T> + Send + 'static) -> Arc<Self> {
        Arc::new(Task {
            status: Mutex::new(Status::Queued(Box::new(job))),
            done: Condvar::new(),
            then: Mutex::new(Vec::new()),
        })
    }

    /// The result, or `None` while the work has not finished.
    pub fn poll(&self) -> Option<Option<T>> {
        match &*lock(&self.status) {
            Status::Done(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// The result if the work finishes within `limit`, which is left to the pool.
    pub fn wait_for(&self, limit: Duration) -> Option<Option<T>> {
        let deadline = Instant::now() + limit;
        let mut status = lock(&self.status);
        loop {
            if let Status::Done(v) = &*status {
                return Some(v.clone());
            }
            let left = deadline.checked_duration_since(Instant::now())?;
            status = self
                .done
                .wait_timeout(status, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// The result, waiting for it; work no thread has started yet runs on
    /// the calling thread instead of waiting for a free one.
    pub fn wait(&self) -> Option<T> {
        let mut status = lock(&self.status);
        loop {
            match std::mem::replace(&mut *status, Status::Running) {
                Status::Queued(job) => {
                    drop(status);
                    return self.finish(job);
                }
                Status::Running => {
                    status = self.done.wait(status).unwrap_or_else(|e| e.into_inner())
                }
                Status::Done(v) => {
                    *status = Status::Done(v.clone());
                    return v;
                }
            }
        }
    }

    /// [`Task::wait`] for a command processor that cannot go on without the
    /// result; while it waits, [`blocked`] is true.
    pub fn wait_blocking(&self) -> Option<T> {
        BLOCKED.fetch_add(1, Ordering::Relaxed);
        let value = self.wait();
        BLOCKED.fetch_sub(1, Ordering::Relaxed);
        value
    }

    fn finish(&self, job: Job<T>) -> Option<T> {
        let value = catch_unwind(AssertUnwindSafe(job)).unwrap_or(None);
        let then = {
            let mut status = lock(&self.status);
            *status = Status::Done(value.clone());
            std::mem::take(&mut *lock(&self.then))
        };
        self.done.notify_all();
        for f in then {
            f();
        }
        value
    }
}

impl<T: Clone + Send + 'static> Dependency for Task<T> {
    fn on_done(&self, f: Box<dyn FnOnce() + Send>) {
        let status = lock(&self.status);
        if matches!(*status, Status::Done(_)) {
            drop(status);
            f();
        } else {
            lock(&self.then).push(f);
        }
    }
}

impl<T: Clone + Send + 'static> Run for Task<T> {
    fn run(&self) {
        let mut status = lock(&self.status);
        if matches!(*status, Status::Queued(_))
            && let Status::Queued(job) = std::mem::replace(&mut *status, Status::Running)
        {
            drop(status);
            self.finish(job);
        }
    }
}

struct Pool {
    queue: Mutex<VecDeque<Arc<dyn Run>>>,
    work: Condvar,
}

impl Pool {
    fn push(&self, task: Arc<dyn Run>) {
        lock(&self.queue).push_back(task);
        self.work.notify_one();
    }

    fn move_to_front(&self, task: *const ()) {
        let mut queue = lock(&self.queue);
        if let Some(i) = queue
            .iter()
            .position(|t| std::ptr::eq(Arc::as_ptr(t).cast::<()>(), task))
            && let Some(t) = queue.remove(i)
        {
            queue.push_front(t);
        }
    }

    fn push_front(&self, task: Arc<dyn Run>) {
        lock(&self.queue).push_front(task);
        self.work.notify_one();
    }

    fn worker(&self) {
        loop {
            let task = {
                let mut queue = lock(&self.queue);
                loop {
                    match queue.pop_front() {
                        // Nobody holds the result any more (caches were cleared).
                        Some(t) if Arc::strong_count(&t) == 1 => continue,
                        Some(t) => break t,
                        None => queue = self.work.wait(queue).unwrap_or_else(|e| e.into_inner()),
                    }
                }
            };
            objc2::rc::autoreleasepool(|_| task.run());
        }
    }
}

/// Compile threads: `PS5SLOP_COMPILE_THREADS`, or one per performance core
/// (Metal's compiler scales about linearly with them).
fn threads() -> usize {
    std::env::var("PS5SLOP_COMPILE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(performance_cores)
        .max(1)
}

fn performance_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        let mut n: libc::c_int = 0;
        let mut len = std::mem::size_of::<libc::c_int>();
        // SAFETY: a NUL-terminated name and an int-sized output buffer.
        let r = unsafe {
            libc::sysctlbyname(
                c"hw.perflevel0.physicalcpu".as_ptr(),
                (&mut n as *mut libc::c_int).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if r == 0 && n > 0 {
            return n as usize;
        }
    }
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            queue: Mutex::new(VecDeque::new()),
            work: Condvar::new(),
        }));
        for n in 0..threads() {
            std::thread::Builder::new()
                .name(format!("gpu compile {n}"))
                .spawn(|| pool.worker())
                .expect("spawning a compile thread");
        }
        pool
    })
}

/// Work queued and not yet started.
pub fn queued() -> usize {
    lock(&pool().queue).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn a_spawned_task_yields_its_result() {
        let task = Task::spawn(|| Some(7));
        assert_eq!(task.wait(), Some(7));
        assert_eq!(task.poll(), Some(Some(7)));
    }

    #[test]
    fn poll_reports_unfinished_work_as_none() {
        let (go, gate) = mpsc::channel::<()>();
        let task = Task::spawn(move || {
            gate.recv().ok();
            Some(1)
        });
        assert_eq!(task.poll(), None);
        go.send(()).unwrap();
        assert_eq!(task.wait(), Some(1));
    }

    #[test]
    fn wait_for_gives_up_on_work_that_takes_longer() {
        let (go, gate) = mpsc::channel::<()>();
        let task = Task::spawn(move || {
            gate.recv().ok();
            Some(3)
        });
        assert_eq!(task.wait_for(Duration::from_millis(20)), None);
        go.send(()).unwrap();
        assert_eq!(task.wait_for(Duration::from_secs(10)), Some(Some(3)));
    }

    #[test]
    fn failed_work_is_done_with_no_value() {
        let task: Arc<Task<u32>> = Task::spawn(|| None);
        assert_eq!(task.wait(), None);
        assert_eq!(task.poll(), Some(None));
    }

    #[test]
    fn a_blocking_wait_counts_as_blocked_while_it_waits() {
        let task = Task::spawn(|| {
            let start = Instant::now();
            while !blocked() && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(1));
            }
            Some(blocked())
        });
        assert_eq!(task.wait_blocking(), Some(true));
    }

    #[test]
    fn a_panicking_job_fails_instead_of_hanging_waiters() {
        let task: Arc<Task<u32>> = Task::spawn(|| panic!("job failed"));
        assert_eq!(task.wait(), None);
    }

    #[test]
    fn a_dependent_task_starts_only_after_its_dependencies() {
        let (go, gate) = mpsc::channel::<()>();
        let first = Task::spawn(move || {
            gate.recv_timeout(Duration::from_secs(10)).ok();
            Some(4)
        });
        let after = {
            let f = first.clone();
            Task::spawn_after(&[&*first], move || f.poll().flatten().map(|v| v + 1))
        };
        assert_eq!(after.wait_for(Duration::from_millis(50)), None);
        go.send(()).unwrap();
        assert_eq!(after.wait_for(Duration::from_secs(10)), Some(Some(5)));
        let done = Task::spawn_after(&[&*first], || Some(9));
        assert_eq!(done.wait_for(Duration::from_secs(10)), Some(Some(9)));
    }

    #[test]
    fn hurried_work_runs_before_older_queued_work() {
        let (go, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let order = Arc::new(Mutex::new(Vec::new()));
        let started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let blockers: Vec<_> = (0..threads())
            .map(|_| {
                let gate = gate.clone();
                let started = started.clone();
                Task::spawn(move || {
                    started.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    gate.lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(10))
                        .ok();
                    Some(0)
                })
            })
            .collect();
        let deadline = Instant::now() + Duration::from_secs(10);
        while started.load(std::sync::atomic::Ordering::SeqCst) < threads()
            && Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        let queued: Vec<_> = (1..=3)
            .map(|n| {
                let order = order.clone();
                Task::spawn(move || {
                    order.lock().unwrap().push(n);
                    Some(n)
                })
            })
            .collect();
        queued[2].hurry();
        for _ in &blockers {
            go.send(()).ok();
        }
        for t in blockers.iter().chain(&queued) {
            t.wait_for(Duration::from_secs(10));
        }
        assert_eq!(order.lock().unwrap().first(), Some(&3));
    }

    #[test]
    fn a_task_may_wait_for_another_while_every_worker_is_busy() {
        let (go, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let blockers: Vec<_> = (0..threads())
            .map(|_| {
                let gate = gate.clone();
                Task::spawn(move || {
                    gate.lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(10))
                        .ok();
                    Some(0)
                })
            })
            .collect();
        let inner = Task::spawn(|| Some(2));
        let outer = Task::spawn(move || inner.wait().map(|v| v * 10));
        assert_eq!(outer.wait(), Some(20));
        for _ in &blockers {
            go.send(()).ok();
        }
        for b in blockers {
            b.wait();
        }
    }
}
