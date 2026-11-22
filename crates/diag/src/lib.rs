//! Logging set-up and rate-limited reports, shared by every crate.

mod limit;

pub use limit::{Distinct, FirstN};

use std::io::IsTerminal;
use tracing_subscriber::EnvFilter;

/// Errors and warnings, the run's milestones, and the targets whose output a
/// switch or a control-socket command asks for. `RUST_LOG` is applied on top.
pub const DEFAULT_FILTER: &str = "warn,ps5slop=info,gpu::trace=info,gpu::snap=info,gpu::probe=info,gpu::watch=info,jit::probe=info";

/// Sends log output to stderr, filtered by [`DEFAULT_FILTER`] and then `RUST_LOG`.
pub fn init_logging() {
    let spec = match std::env::var("RUST_LOG") {
        Ok(user) => format!("{DEFAULT_FILTER},{user}"),
        Err(_) => DEFAULT_FILTER.to_owned(),
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::builder().parse_lossy(spec))
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .without_time()
        .try_init();
}
