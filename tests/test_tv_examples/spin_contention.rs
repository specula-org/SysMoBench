// Execute the patched acquisition/release methods with a deterministic CAS race.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

static FAILED_CAS: Barrier = Barrier::new(2);
static RESUME_WAITER: Barrier = Barrier::new(2);
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct ObservedLock(AtomicBool);

impl ObservedLock {
    fn compare_exchange(
        &self,
        current: bool,
        new: bool,
        success: Ordering,
        failure: Ordering,
    ) -> Result<bool, bool> {
        let result = self.0.compare_exchange(current, new, success, failure);
        if result.is_err() {
            // Pause immediately after the failed atomic operation so the holder
            // can release and finish logging before the waiter retries.
            FAILED_CAS.wait();
            RESUME_WAITER.wait();
        }
        result
    }

    fn store(&self, value: bool, order: Ordering) {
        self.0.store(value, order);
    }
}

struct SpinLockInner {
    lock: ObservedLock,
}

struct SpinTrace {
    inner: SpinLockInner,
}

impl SpinTrace {
    // SPIN_TRACE_METHODS
}

fn trace_event(action: &str, _address: usize, actor: usize, held: bool) {
    let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let state = if held { "locked" } else { "unlocked" };
    println!(
        "{{\"seq\":{seq},\"action\":\"{action}\",\"actor\":{actor},\"state\":\"{state}\",\"lock\":0}}"
    );
}

fn main() {
    let lock = Arc::new(SpinTrace {
        inner: SpinLockInner {
            lock: ObservedLock(AtomicBool::new(false)),
        },
    });
    trace_event("TryAcquireBlocking", 0, 0, false);
    lock.acquire_lock_with_id(0);
    let waiter_lock = Arc::clone(&lock);
    let waiter = std::thread::spawn(move || {
        trace_event("TryAcquireBlocking", 0, 1, true);
        waiter_lock.acquire_lock_with_id(1);
        waiter_lock.release_lock(1);
    });
    FAILED_CAS.wait();
    lock.release_lock(0);
    RESUME_WAITER.wait();
    waiter.join().unwrap();
}
