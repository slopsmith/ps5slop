//! Work that runs on a pool of compile threads, so the command processor
//! does not stall while Metal compiles programs and pipelines.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

type Job<T> = Box<dyn FnOnce() -> Option<T> + Send>;

enum Status<T> {
    Queued(Job<T>),
    Running,
    Done(Option<T>),
}

/// A result being computed on the pool; `None` inside means the work failed.
pub struct Task<T> {
    status: Mutex<Status<T>>,
    done: Condvar,
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
        let task = Arc::new(Task {
            status: Mutex::new(Status::Queued(Box::new(job))),
            done: Condvar::new(),
        });
        pool().push(task.clone());
        task
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
            status = self.done.wait_timeout(status, left).unwrap_or_else(|e| e.into_inner()).0;
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
                Status::Running => status = self.done.wait(status).unwrap_or_else(|e| e.into_inner()),
                Status::Done(v) => {
                    *status = Status::Done(v.clone());
                    return v;
                }
            }
        }
    }

    fn finish(&self, job: Job<T>) -> Option<T> {
        let value = catch_unwind(AssertUnwindSafe(job)).unwrap_or(None);
        *lock(&self.status) = Status::Done(value.clone());
        self.done.notify_all();
        value
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

/// Compile threads: `PS5SLOP_COMPILE_THREADS`, or 4.
fn threads() -> usize {
    std::env::var("PS5SLOP_COMPILE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1)
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
    fn a_panicking_job_fails_instead_of_hanging_waiters() {
        let task: Arc<Task<u32>> = Task::spawn(|| panic!("job failed"));
        assert_eq!(task.wait(), None);
    }

    #[test]
    fn a_task_may_wait_for_another_while_every_worker_is_busy() {
        let (go, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let blockers: Vec<_> = (0..threads())
            .map(|_| {
                let gate = gate.clone();
                Task::spawn(move || {
                    gate.lock().unwrap().recv_timeout(Duration::from_secs(10)).ok();
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
