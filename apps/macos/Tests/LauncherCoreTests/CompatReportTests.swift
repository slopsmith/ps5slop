import Foundation
import Testing

@testable import LauncherCore

struct CompatReportTests {
    private func report(_ answers: [CompatReport.Question: Bool]) -> CompatReport {
        var report = CompatReport(
            titleId: "PPSA21402", gameName: "Balatro", gameVersion: "01.015.000",
            macos: "27.3", chip: "Apple M4 Pro", ps5slop: "493e012")
        report.answers = answers
        return report
    }

    private func query(_ url: URL?) -> [String: String] {
        let items = URLComponents(url: url!, resolvingAgainstBaseURL: false)!.queryItems!
        return Dictionary(uniqueKeysWithValues: items.map { ($0.name, $0.value ?? "") })
    }

    @Test func asksOnlyWhileEachStepWorked() {
        #expect(report([:]).askedQuestions == [.onScreen])
        #expect(report([.onScreen: true, .menus: false]).askedQuestions == [.onScreen, .menus])
        #expect(report([.onScreen: true, .menus: true, .gameplay: true, .playedThrough: true]).askedQuestions == CompatReport.Question.allCases)
    }

    @Test func stepsThroughTheAskedQuestions() {
        let r = report([.onScreen: true, .menus: true, .gameplay: false, .playedThrough: true])
        #expect(r.question(after: .menus) == .gameplay)
        #expect(r.question(after: .gameplay) == nil)
        #expect(r.question(before: .gameplay) == .menus)
        #expect(r.question(before: .onScreen) == nil)
        let through = report([.onScreen: true, .menus: true, .gameplay: true, .playedThrough: true])
        #expect(through.question(after: .playedThrough) == .broken)
        #expect(through.question(after: .playedFully) == nil)
    }

    @Test func givesTheStatusTheCompatRepoChecksFor() {
        #expect(report([:]).status == nil)
        #expect(report([.onScreen: false]).status == .nothing)
        #expect(report([.onScreen: true, .menus: false]).status == .boots)
        #expect(report([.onScreen: true, .menus: true, .gameplay: false]).status == .menus)
        #expect(report([.onScreen: true, .menus: true, .gameplay: true, .playedThrough: false]).status == .ingame)
        let through: [CompatReport.Question: Bool] = [.onScreen: true, .menus: true, .gameplay: true, .playedThrough: true]
        #expect(report(through).status == nil)
        #expect(report(through.merging([.broken: true, .playedFully: true]) { $1 }).status == .playable)
        #expect(report(through.merging([.broken: false, .playedFully: false]) { $1 }).status == .playable)
        #expect(report(through.merging([.broken: false, .playedFully: true]) { $1 }).status == .perfect)
    }

    @Test func ignoresAnswersToQuestionsNoLongerAsked() {
        let r = report([.onScreen: true, .menus: false, .gameplay: true, .playedThrough: true, .broken: false, .playedFully: true])
        #expect(r.status == .boots)
        let q = query(r.url)
        #expect(q["gameplay"] == "no")
        #expect(q["played_fully"] == "no")
    }

    @Test func fillsInEveryFieldOfTheLauncherForm() {
        var r = report([.onScreen: true, .menus: true, .gameplay: false])
        r.notes = "  Crashes after the intro + logo.\nSee log.  "
        #expect(r.url!.absoluteString.hasPrefix("https://github.com/slopsmith/ps5slop-compat/issues/new?"))
        #expect(!r.url!.absoluteString.contains("+"))
        #expect(
            query(r.url) == [
                "template": "launcher-report.yml",
                "title_id": "PPSA21402", "game_name": "Balatro", "game_version": "01.015.000",
                "on_screen": "yes", "menus": "yes", "gameplay": "no",
                "played_through": "no", "broken": "no", "played_fully": "no",
                "notes": "Crashes after the intro + logo.\nSee log.",
                "macos": "27.3", "chip": "Apple M4 Pro", "ps5slop": "493e012",
            ])
    }

    @Test func leavesOutEmptyOptionalFields() {
        var r = report([.onScreen: false])
        r.gameName = nil
        r.gameVersion = nil
        r.notes = " \n"
        let q = query(r.url)
        #expect(q["game_name"] == nil)
        #expect(q["game_version"] == nil)
        #expect(q["notes"] == nil)
    }

    @Test func hasNoLinkUntilEveryAskedQuestionIsAnswered() {
        #expect(report([.onScreen: true]).url == nil)
    }
}
