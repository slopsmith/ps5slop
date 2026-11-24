import Foundation
import Testing

@testable import LauncherCore

struct UpdateCheckTests {
    private let a = String(repeating: "a", count: 40)
    private let b = String(repeating: "b", count: 40)

    private func nightly(_ commit: String) -> Nightly {
        Nightly(
            commit: commit, date: "2026-11-24",
            download: URL(string: "https://example.com/app.zip")!,
            release: URL(string: "https://example.com/nightly")!)
    }

    @Test func offersTheNightlyWhenItsCommitDiffers() {
        #expect(UpdateCheck.status(current: a, nightly: nightly(b)) == .available(nightly(b)))
    }

    @Test func isUpToDateOnTheNightlysCommit() {
        #expect(UpdateCheck.status(current: a, nightly: nightly(a)) == .upToDate)
    }

    @Test func neverOffersUpdatesToLocalBuilds() {
        for commit in [nil, "", "\(a)-dirty", "\(a)-dev", "unknown"] {
            #expect(UpdateCheck.status(current: commit, nightly: nightly(b)) == .localBuild)
        }
    }

    @Test func treatsOnlyFullCommitsAsReleaseBuilds() {
        #expect(UpdateCheck.isRelease(a))
        #expect(!UpdateCheck.isRelease(String(a.dropLast())))
        #expect(!UpdateCheck.isRelease(a.uppercased()))
        #expect(!UpdateCheck.isRelease("\(a)-dev"))
    }

    @Test func decodesTheFeedAndIgnoresUnknownFields() throws {
        let json = """
            {"commit":"\(b)","date":"2026-11-24T03:12:00Z","download":"https://example.com/app.zip",
             "release":"https://example.com/nightly","extra":1}
            """
        let feed = try Nightly.decode(Data(json.utf8))
        #expect(feed.commit == b)
        #expect(feed.download == URL(string: "https://example.com/app.zip"))
    }

    @Test func rejectsAFeedWithoutACommit() {
        let json = #"{"date":"2026-11-24","download":"https://example.com/a.zip","release":"https://example.com/n"}"#
        #expect(throws: DecodingError.self) { try Nightly.decode(Data(json.utf8)) }
    }

    @Test func readsTheFeedFromTheRollingNightlyRelease() {
        let feed = UpdateCheck.feed(source: URL(string: "https://github.com/slopsmith/ps5slop")!)
        #expect(feed.absoluteString == "https://github.com/slopsmith/ps5slop/releases/download/nightly/nightly.json")
    }
}
