import Foundation

/// The launcher's files under `~/Library/Application Support/ps5slop/launcher`:
/// the library folders and the last `info` result, shown at once on the next start.
public struct LibraryStore: Sendable {
    public let root: URL

    public init(root: URL) {
        self.root = root
    }

    public static var standard: LibraryStore {
        LibraryStore(root: Paths.supportFolder.appendingPathComponent("launcher", isDirectory: true))
    }

    public var iconsFolder: URL { root.appendingPathComponent("icons", isDirectory: true) }
    var foldersFile: URL { root.appendingPathComponent("library.json") }
    var cacheFile: URL { root.appendingPathComponent("games-cache.json") }

    private struct FolderList: Codable {
        var folders: [String]
    }

    public func loadFolders() -> [URL] {
        guard let data = try? Data(contentsOf: foldersFile),
            let list = try? JSONDecoder().decode(FolderList.self, from: data)
        else { return [] }
        return list.folders.map { URL(fileURLWithPath: $0) }
    }

    public func saveFolders(_ folders: [URL]) throws {
        let list = FolderList(folders: folders.map(\.standardizedFileURL.path))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        try write(encoder.encode(list), to: foldersFile)
    }

    public func loadCache() -> [GameEntry] {
        guard let data = try? Data(contentsOf: cacheFile) else { return [] }
        return (try? GameInfoDecoder.decode(data)) ?? []
    }

    public func saveCache(_ games: [GameEntry]) throws {
        try write(GameInfoDecoder.encode(games), to: cacheFile)
    }

    private func write(_ data: Data, to file: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try data.write(to: file, options: .atomic)
    }
}

/// Adds `new` to `folders`, skipping duplicates and paths already listed.
public func addingFolders(_ new: [URL], to folders: [URL]) -> [URL] {
    var result = folders
    for url in new {
        let std = url.standardizedFileURL
        if !result.contains(where: { $0.standardizedFileURL.path == std.path }) {
            result.append(std)
        }
    }
    return result
}

public enum Paths {
    public static var supportFolder: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/ps5slop", isDirectory: true)
    }

    public static var capturesFolder: URL {
        supportFolder.appendingPathComponent("captures", isDirectory: true)
    }

    public static var defaultFontsFolder: URL {
        supportFolder.appendingPathComponent("fonts", isDirectory: true)
    }
}
