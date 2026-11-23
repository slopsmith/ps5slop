import Foundation
import LauncherCore
import Observation

@MainActor
@Observable
final class Library {
    enum Source {
        case store(LibraryStore)
        /// A fixed `info` JSON file: nothing is scanned or saved.
        case fixture(URL)
    }

    private(set) var folders: [URL] = []
    private(set) var games: [GameEntry] = []
    private(set) var isScanning = false
    private(set) var scanError: String?

    private let source: Source
    private var scanTask: Task<Void, Never>?

    init(source: Source) {
        self.source = source
        switch source {
        case .store(let store):
            folders = store.loadFolders()
            games = store.loadCache()
        case .fixture(let url):
            do {
                games = try GameInfoDecoder.decode(Data(contentsOf: url))
                folders = Self.commonFolders(of: games)
            } catch {
                scanError = "Cannot read \(url.path): \(error.localizedDescription)"
            }
        }
    }

    func add(_ urls: [URL]) {
        let updated = addingFolders(urls, to: folders)
        guard updated != folders else { return }
        folders = updated
        saveFolders()
        refresh()
    }

    func remove(_ folder: URL) {
        folders.removeAll { $0 == folder }
        games.removeAll { $0.isInside(folder) }
        saveFolders()
        saveCache()
    }

    func refresh() {
        guard case .store(let store) = source else { return }
        guard let binary = BinaryLocator.ps5slop() else {
            scanError = "The ps5slop binary was not found. Set PS5SLOP_BIN or build target/release/ps5slop."
            return
        }
        scanTask?.cancel()
        let runner = InfoRunner(binary: binary, iconsFolder: store.iconsFolder)
        let paths = folders
        isScanning = true
        scanTask = Task {
            do {
                let found = try await runner.scan(paths)
                guard !Task.isCancelled else { return }
                games = found
                scanError = nil
                saveCache()
            } catch {
                guard !Task.isCancelled else { return }
                scanError = error.localizedDescription
            }
            isScanning = false
        }
    }

    private func saveFolders() {
        guard case .store(let store) = source else { return }
        do {
            try store.saveFolders(folders)
        } catch {
            scanError = "Cannot save the folder list: \(error.localizedDescription)"
        }
    }

    private func saveCache() {
        guard case .store(let store) = source else { return }
        try? store.saveCache(games)
    }

    private static func commonFolders(of games: [GameEntry]) -> [URL] {
        let parents = games.map { URL(fileURLWithPath: $0.path).deletingLastPathComponent() }
        return addingFolders(parents, to: [])
    }
}
