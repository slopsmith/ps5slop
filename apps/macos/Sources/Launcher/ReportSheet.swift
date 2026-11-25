import AppKit
import LauncherCore
import SwiftUI

@MainActor
@Observable
final class Reporting {
    var game: GameEntry?
    var answers: [CompatReport.Question: Bool] = [:]

    func start(_ game: GameEntry, answers: [CompatReport.Question: Bool] = [:]) {
        self.answers = answers
        self.game = game
    }
}

struct ReportSheet: View {
    private enum Step: Hashable {
        case question(CompatReport.Question)
        case summary
    }

    let game: GameEntry
    @State private var report: CompatReport
    @State private var step: Step
    @State private var forward = true
    @Environment(\.dismiss) private var dismiss

    init?(game: GameEntry, answers: [CompatReport.Question: Bool] = [:]) {
        guard let titleId = game.titleId else { return nil }
        var report = CompatReport(
            titleId: titleId, gameName: game.name, gameVersion: game.version,
            macos: CompatReport.localMacOS, chip: CompatList.localChip ?? "unknown",
            ps5slop: BuildInfo.current.shortCommit)
        report.answers = answers
        self.game = game
        _report = State(initialValue: report)
        let open = report.askedQuestions.first { report.answers[$0] == nil }
        _step = State(initialValue: open.map(Step.question) ?? .summary)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ZStack {
                switch step {
                case .question(let question):
                    QuestionPage(question: question, answer: report.answers[question]) { answer($0, to: question) }
                        .id(question)
                        .transition(slide)
                case .summary:
                    SummaryPage(report: $report, change: { go(to: .question($0), forward: false) })
                        .transition(slide)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .clipped()
            Divider()
            footer
        }
        .frame(width: 560, height: 580)
    }

    private var slide: AnyTransition {
        .asymmetric(
            insertion: .move(edge: forward ? .trailing : .leading).combined(with: .opacity),
            removal: .move(edge: forward ? .leading : .trailing).combined(with: .opacity))
    }

    private var header: some View {
        HStack(spacing: 12) {
            GameIcon(game: game, size: 40)
            VStack(alignment: .leading, spacing: 2) {
                Text("Report Compatibility")
                    .font(.headline)
                Text(game.displayName)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            progress
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var progress: some View {
        let asked = report.askedQuestions
        let current: Int? = if case .question(let q) = step { asked.firstIndex(of: q) } else { nil }
        return HStack(spacing: 5) {
            ForEach(CompatReport.Question.allCases, id: \.self) { question in
                let index = asked.firstIndex(of: question)
                Capsule()
                    .fill(fill(index: index, current: current))
                    .frame(width: index != nil && index == current ? 18 : 8, height: 8)
            }
        }
        .animation(.spring(duration: 0.3), value: step)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(current.map { "Question \($0 + 1)" } ?? "Summary")
    }

    private func fill(index: Int?, current: Int?) -> AnyShapeStyle {
        guard let index else { return AnyShapeStyle(.quaternary) }
        if index == current { return AnyShapeStyle(Color.accentColor) }
        if current == nil || index < current! { return AnyShapeStyle(Color.accentColor.opacity(0.45)) }
        return AnyShapeStyle(.quaternary)
    }

    private var footer: some View {
        HStack {
            Button("Back", systemImage: "chevron.left", action: back)
                .disabled(step == .question(.onScreen))
            Spacer()
            Button("Cancel", role: .cancel) { dismiss() }
                .keyboardShortcut(.cancelAction)
            if step == .summary {
                Button("Open on GitHub", systemImage: "arrow.up.forward.app") {
                    if let url = report.url { NSWorkspace.shared.open(url) }
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(report.url == nil)
            }
        }
        .controlSize(.large)
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private func answer(_ value: Bool, to question: CompatReport.Question) {
        report.answers[question] = value
        go(to: report.question(after: question).map(Step.question) ?? .summary, forward: true)
    }

    private func back() {
        switch step {
        case .question(let question):
            if let previous = report.question(before: question) { go(to: .question(previous), forward: false) }
        case .summary:
            if let last = report.askedQuestions.last { go(to: .question(last), forward: false) }
        }
    }

    private func go(to next: Step, forward: Bool) {
        self.forward = forward
        withAnimation(.spring(duration: 0.35)) { step = next }
    }
}

private struct QuestionPage: View {
    let question: CompatReport.Question
    let answer: Bool?
    let choose: (Bool) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Spacer(minLength: 0)
            Text("Question \(question.rawValue + 1)")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(question.title)
                .font(.title2.weight(.semibold))
                .fixedSize(horizontal: false, vertical: true)
            Text(question.detail)
                .font(.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 12) {
                choice("Yes", value: true, shortcut: "y")
                choice("No", value: false, shortcut: "n")
            }
            .padding(.top, 10)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private func choice(_ title: String, value: Bool, shortcut: KeyEquivalent) -> some View {
        let label = Text(title).frame(maxWidth: .infinity)
        if answer == value {
            Button { choose(value) } label: { label }
                .buttonStyle(.borderedProminent)
                .controlSize(.extraLarge)
                .keyboardShortcut(shortcut, modifiers: [])
                .accessibilityValue("Your earlier answer")
        } else {
            Button { choose(value) } label: { label }
                .buttonStyle(.bordered)
                .controlSize(.extraLarge)
                .keyboardShortcut(shortcut, modifiers: [])
        }
    }
}

private struct SummaryPage: View {
    @Binding var report: CompatReport
    let change: (CompatReport.Question) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                if let status = report.status {
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        CompatBadge(status: status, showsSymbol: true)
                            .font(.title3)
                        Text(status.meaning)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                section("Your Answers") {
                    VStack(spacing: 0) {
                        ForEach(report.askedQuestions, id: \.self) { question in
                            Button { change(question) } label: {
                                HStack {
                                    Text(question.title)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                    Text(report.answers[question] == true ? "Yes" : "No")
                                        .foregroundStyle(.secondary)
                                    Image(systemName: "chevron.right")
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.tertiary)
                                }
                                .padding(.vertical, 7)
                                .contentShape(.rect)
                            }
                            .buttonStyle(.plain)
                            .help("Change this answer")
                            if question != report.askedQuestions.last { Divider() }
                        }
                    }
                }
                section("Notes") {
                    TextField(
                        "Notes", text: $report.notes,
                        prompt: Text("What works, where it stops, what looks or sounds wrong, the frame rate"),
                        axis: .vertical
                    )
                    .lineLimit(3...6)
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
                }
                section("Sent with the Report") {
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                        row("Title ID", report.titleId)
                        row("Game version", report.gameVersion ?? "Unknown")
                        row("Mac", "\(report.chip), macOS \(report.macos)")
                        row("ps5slop", report.ps5slop)
                    }
                    .font(.callout)
                    Text("GitHub opens with the report filled in. Check it there and submit it; you need a GitHub account.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.top, 4)
                }
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 20)
        }
    }

    private func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content()
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).textSelection(.enabled)
        }
    }
}
