//! Finds games in folders and reads what a launcher shows of them: name,
//! title id, version and icon.

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

use pkg::Package;

/// How deep a scan descends below each folder it is given.
const MAX_DEPTH: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Pkg,
    Folder,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    pub path: PathBuf,
    pub kind: Kind,
    pub title_id: Option<String>,
    pub name: Option<String>,
    pub version: Option<String>,
    pub icon: Option<PathBuf>,
    pub error: Option<String>,
}

impl Game {
    pub fn to_json(&self) -> serde_json::Value {
        let kind = match self.kind {
            Kind::Pkg => "pkg",
            Kind::Folder => "folder",
        };
        serde_json::json!({
            "path": self.path,
            "kind": kind,
            "title_id": self.title_id,
            "name": self.name,
            "version": self.version,
            "icon": self.icon,
            "error": self.error,
        })
    }
}

/// Every game in `paths`: each is a `.pkg`, an app folder (holding
/// `eboot.bin`) or a folder searched for those. Icons are written to `icons`.
pub fn scan(paths: &[PathBuf], icons: Option<&Path>) -> Vec<Game> {
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    for p in paths {
        walk(p, 0, &mut seen, &mut found);
    }
    found.iter().map(|(p, k)| read(p, *k, icons)).collect()
}

fn walk(path: &Path, depth: usize, seen: &mut HashSet<PathBuf>, found: &mut Vec<(PathBuf, Kind)>) {
    let Ok(real) = std::fs::canonicalize(path) else {
        return;
    };
    if !seen.insert(real) {
        return;
    }
    if path.is_file() {
        if path
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("pkg"))
        {
            found.push((path.to_path_buf(), Kind::Pkg));
        }
        return;
    }
    if path.join("eboot.bin").is_file() {
        found.push((path.to_path_buf(), Kind::Folder));
        return;
    }
    if depth >= MAX_DEPTH {
        return;
    }
    let Ok(dir) = std::fs::read_dir(path) else {
        return;
    };
    let mut entries: Vec<PathBuf> = dir
        .flatten()
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| e.path())
        .collect();
    entries.sort();
    for e in entries {
        walk(&e, depth + 1, seen, found);
    }
}

fn read(path: &Path, kind: Kind, icons: Option<&Path>) -> Game {
    let mut game = Game {
        path: std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()),
        kind,
        title_id: None,
        name: None,
        version: None,
        icon: None,
        error: None,
    };
    let icon = match kind {
        Kind::Pkg => match Package::open(path) {
            Ok(package) => {
                game.title_id = package.title_id();
                (game.name, game.version) = package.name_and_version();
                package
                    .sce_sys_entries()
                    .into_iter()
                    .find(|(name, _)| *name == "icon0.png")
                    .and_then(|(_, e)| package.read_entry(&e).ok())
            }
            Err(e) => {
                game.error = Some(format!("cannot open the package: {e}"));
                None
            }
        },
        Kind::Folder => {
            let sce_sys = path.join("sce_sys");
            if let Ok(j) = std::fs::read(sce_sys.join("param.json")) {
                game.title_id = pkg::title::from_param_json(&j);
                (game.name, game.version) = pkg::title::name_from_param_json(&j);
            } else if let Ok(s) = std::fs::read(sce_sys.join("param.sfo")) {
                game.title_id = pkg::title::from_param_sfo(&s);
                (game.name, game.version) = pkg::title::name_from_param_sfo(&s);
            }
            std::fs::read(sce_sys.join("icon0.png")).ok()
        }
    };
    if let (Some(dir), Some(png)) = (icons, icon) {
        game.icon = save_icon(dir, &game, &png);
    }
    game
}

/// Writes `png` to `dir` under a name that changes with the game's path and
/// the icon's bytes, so a replaced game gets a new file.
fn save_icon(dir: &Path, game: &Game, png: &[u8]) -> Option<PathBuf> {
    let mut h = DefaultHasher::new();
    game.path.hash(&mut h);
    png.hash(&mut h);
    let stem = game.title_id.as_deref().unwrap_or("game");
    let file = dir.join(format!("{stem}-{:016x}.png", h.finish()));
    if !file.is_file() {
        std::fs::create_dir_all(dir).ok()?;
        std::fs::write(&file, png).ok()?;
    }
    Some(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_app_folders_and_packages() {
        let root = std::env::temp_dir().join(format!("ps5slop-library-{}", std::process::id()));
        let app = root.join("Games/My Game");
        std::fs::create_dir_all(app.join("sce_sys")).unwrap();
        std::fs::write(app.join("eboot.bin"), b"").unwrap();
        std::fs::write(
            app.join("sce_sys/param.json"),
            br#"{"titleId":"ABCD12345","contentVersion":"01.000.000",
                "localizedParameters":{"defaultLanguage":"en-US","en-US":{"titleName":"My Game"}}}"#,
        )
        .unwrap();
        std::fs::write(app.join("sce_sys/icon0.png"), b"png").unwrap();
        std::fs::create_dir_all(app.join("nested")).unwrap();
        std::fs::write(app.join("nested/eboot.bin"), b"").unwrap();
        std::fs::write(root.join("Games/broken.pkg"), b"not a package").unwrap();
        let icons = root.join("icons");

        let games = scan(&[root.join("Games")], Some(&icons));
        std::fs::remove_dir_all(&root).unwrap();

        assert_eq!(games.len(), 2, "app folders are not searched inside");
        let app = games.iter().find(|g| g.kind == Kind::Folder).unwrap();
        assert_eq!(app.title_id.as_deref(), Some("ABCD12345"));
        assert_eq!(app.name.as_deref(), Some("My Game"));
        assert!(app.icon.as_ref().unwrap().starts_with(&icons));
        let pkg = games.iter().find(|g| g.kind == Kind::Pkg).unwrap();
        assert!(pkg.error.is_some());
    }
}
