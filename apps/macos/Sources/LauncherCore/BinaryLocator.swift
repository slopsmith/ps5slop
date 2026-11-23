import Foundation

public enum BinaryLocator {
    /// The `ps5slop` binary: the copy in the app bundle, else `PS5SLOP_BIN`,
    /// else `target/release/ps5slop` in the current folder or a parent of it.
    public static func ps5slop(
        bundle: Bundle = .main,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        currentDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) -> URL? {
        if let url = bundle.url(forAuxiliaryExecutable: "ps5slop") { return url }
        if let path = environment["PS5SLOP_BIN"], !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return findUpwards("target/release/ps5slop", from: currentDirectory)
    }

    /// `scripts/fetch-fonts`: bundled in Resources, else found from the current folder upwards.
    public static func fetchFontsScript(
        bundle: Bundle = .main,
        currentDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) -> URL? {
        bundle.url(forResource: "fetch-fonts", withExtension: nil)
            ?? findUpwards("scripts/fetch-fonts", from: currentDirectory)
    }

    static func findUpwards(_ relative: String, from start: URL) -> URL? {
        var dir = start.standardizedFileURL
        for _ in 0..<5 {
            let candidate = dir.appendingPathComponent(relative)
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
            let parent = dir.deletingLastPathComponent()
            if parent == dir { break }
            dir = parent
        }
        return nil
    }
}
