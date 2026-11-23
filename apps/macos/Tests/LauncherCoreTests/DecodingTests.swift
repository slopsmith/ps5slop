import Foundation
import Testing

@testable import LauncherCore

struct DecodingTests {
    private func fixture() throws -> [GameEntry] {
        let url = try #require(Bundle.module.url(forResource: "library", withExtension: "json", subdirectory: "Fixtures"))
        return try GameInfoDecoder.decode(Data(contentsOf: url))
    }

    @Test func decodesEveryEntryAndIgnoresUnknownFields() throws {
        let games = try fixture()
        #expect(games.count == 7)
        let rally = try #require(games.first { $0.titleId == "PPSA90002" })
        #expect(rally.kind == .folder)
        #expect(rally.name == "Rustbucket Rally")
        #expect(rally.version == "01.000")
        #expect(rally.compat == Compat(status: .ingame, note: "Made-up title: races run, menus flicker."))
    }

    @Test func decodesEveryCompatStatus() throws {
        let statuses = try fixture().compactMap(\.compat?.status)
        #expect(Set(statuses) == [.perfect, .playable, .ingame, .menus, .boots])
    }

    @Test func keepsUnknownStatusAndKindAsData() throws {
        let json = #"[{"path":"/x","kind":"disc","compat":{"status":"Legendary","note":null}}]"#
        let game = try #require(GameInfoDecoder.decode(Data(json.utf8)).first)
        #expect(game.kind == .unknown)
        #expect(game.compat?.status == .other("Legendary"))
        #expect(game.titleId == nil && game.error == nil)
    }

    @Test func entryWithErrorFallsBackToFileName() throws {
        let broken = try #require(try fixture().first { $0.error != nil })
        #expect(broken.error == "not a PS4 or PS5 package")
        #expect(broken.displayName == "broken")
    }

    @Test func roundTripsThroughTheCacheFormat() throws {
        let games = try fixture()
        #expect(try GameInfoDecoder.decode(GameInfoDecoder.encode(games)) == games)
    }

    @Test func searchMatchesNameAndTitleId() throws {
        let games = try fixture()
        #expect(games.filter { $0.matches("lantern") }.map(\.titleId) == ["PPSA90003"])
        #expect(games.filter { $0.matches("ppsa90004") }.map(\.name) == ["Tidepool Tactics"])
        #expect(games.filter { $0.matches("  ") }.count == games.count)
    }

    @Test func gamesBelongToTheirFolderOnly() {
        let game = GameEntry(path: "/Games/A/x.pkg", kind: .pkg)
        #expect(game.isInside(URL(fileURLWithPath: "/Games/A")))
        #expect(game.isInside(URL(fileURLWithPath: "/Games/A/x.pkg")))
        #expect(!game.isInside(URL(fileURLWithPath: "/Games/AB")))
    }
}
