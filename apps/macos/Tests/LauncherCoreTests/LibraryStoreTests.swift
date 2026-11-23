import Foundation
import Testing

@testable import LauncherCore

struct LibraryStoreTests {
    private let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ps5slop-launcher-tests-\(UUID().uuidString)", isDirectory: true)

    @Test func emptyWhenNothingWasSaved() {
        let store = LibraryStore(root: root)
        #expect(store.loadFolders().isEmpty)
        #expect(store.loadCache().isEmpty)
    }

    @Test func foldersSurviveAReload() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let folders = [URL(fileURLWithPath: "/Volumes/Games"), URL(fileURLWithPath: "/Users/me/pkgs/One.pkg")]
        try LibraryStore(root: root).saveFolders(folders)
        #expect(LibraryStore(root: root).loadFolders() == folders)
    }

    @Test func cacheSurvivesAReload() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let games = [
            GameEntry(path: "/g/a.pkg", kind: .pkg, titleId: "PPSA90001", name: "A", compat: Compat(status: .menus)),
            GameEntry(path: "/g/b", kind: .folder, error: "unreadable"),
        ]
        try LibraryStore(root: root).saveCache(games)
        #expect(LibraryStore(root: root).loadCache() == games)
    }

    @Test func corruptFilesReadAsEmpty() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let store = LibraryStore(root: root)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("{".utf8).write(to: store.foldersFile)
        try Data("nope".utf8).write(to: store.cacheFile)
        #expect(store.loadFolders().isEmpty)
        #expect(store.loadCache().isEmpty)
    }

    @Test func addingFoldersSkipsDuplicates() {
        let a = URL(fileURLWithPath: "/g/a")
        let result = addingFolders([URL(fileURLWithPath: "/g/./a"), URL(fileURLWithPath: "/g/b"), a], to: [a])
        #expect(result.map(\.path) == ["/g/a", "/g/b"])
    }
}

struct LogBufferTests {
    @Test func joinsSplitLinesAndKeepsTheLastOnes() {
        var log = LogBuffer(capacity: 3)
        log.append(Data("one\ntw".utf8))
        log.append(Data("o\nthree\nfour\nfi".utf8))
        #expect(log.lines == ["two", "three", "four"])
        log.flush()
        #expect(log.lines == ["three", "four", "fi"])
    }
}
