import Foundation
import Testing

@testable import LauncherCore

struct CompatListTests {
    private let json = Data(
        #"""
        {"version": 1, "titles": {"PPSA21402": {"name": "Balatro", "status": "playable", "reports": [
          {"issue": 2, "url": "u2", "date": "2026-11-23T00:00:00Z", "status": "playable", "game_version": null,
           "notes": "Music crackled.", "macos": "26.2", "chip": "Apple M2", "ps5slop": "abc"},
          {"issue": 1, "url": "u1", "date": "2026-11-20T00:00:00Z", "status": "perfect", "game_version": "01.015.000",
           "notes": null, "macos": "27.3", "chip": "Apple M4 Pro", "ps5slop": "def"}
        ]}}}
        """#.utf8)

    @Test func prefersTheNewestReportFromTheSameChip() throws {
        let list = try CompatList.decode(json)
        let compat = list.compat(for: "PPSA21402", chip: "Apple M4 Pro")
        #expect(compat == Compat(status: .perfect, note: "Reported on Apple M4 Pro, macOS 27.3, ps5slop def."))
    }

    @Test func fallsBackToTheNewestReport() throws {
        let list = try CompatList.decode(json)
        let compat = list.compat(for: "PPSA21402", chip: "Apple M1")
        #expect(compat == Compat(status: .playable, note: "Music crackled. Reported on Apple M2, macOS 26.2, ps5slop abc."))
    }

    @Test func appliesToGamesByTitleId() throws {
        let games = [
            GameEntry(path: "/a.pkg", kind: .pkg, titleId: "PPSA21402"),
            GameEntry(path: "/b.pkg", kind: .pkg, titleId: "PPSA00001", compat: Compat(status: .boots)),
        ]
        let applied = try CompatList.decode(json).applied(to: games, chip: nil)
        #expect(applied[0].compat?.status == .playable)
        #expect(applied[1].compat == nil)
    }

    @Test func rejectsOtherVersions() {
        #expect(throws: (any Error).self) { try CompatList.decode(Data(#"{"version": 2, "titles": {}}"#.utf8)) }
    }
}
