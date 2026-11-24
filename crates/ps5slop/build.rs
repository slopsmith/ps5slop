//! Stamps the binary with the commit it was built from, as `SLOP_COMMIT`.
//! Release builds pass `SLOP_COMMIT` in; other builds ask git for HEAD and add
//! `-dev`, because the working copy may differ from that commit.

use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    out.status.success().then(|| text.trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-env-changed=SLOP_COMMIT");
    let commit = match std::env::var("SLOP_COMMIT") {
        Ok(commit) if !commit.is_empty() => commit,
        _ => {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            if let Some(head) = git(
                dir,
                &["rev-parse", "--path-format=absolute", "--git-path", "HEAD"],
            ) {
                println!("cargo:rerun-if-changed={head}");
            }
            match git(dir, &["rev-parse", "HEAD"]) {
                Some(sha) => format!("{sha}-dev"),
                None => "unknown".to_owned(),
            }
        }
    };
    println!("cargo:rustc-env=SLOP_COMMIT={commit}");
}
