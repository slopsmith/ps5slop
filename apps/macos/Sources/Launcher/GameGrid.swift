import LauncherCore
import SwiftUI

struct GameGrid: View {
    let games: [GameEntry]
    @Binding var selection: GameEntry.ID?
    let runningID: GameEntry.ID?
    let play: (GameEntry) -> Void

    var body: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 156, maximum: 200), spacing: 22)], spacing: 26) {
                ForEach(games) { game in
                    GameCard(
                        game: game, selected: selection == game.id, running: runningID == game.id,
                        select: { selection = game.id }, play: { play(game) })
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
        }
        .contentMargins(.bottom, 72, for: .scrollContent)
    }
}

struct GameCard: View {
    let game: GameEntry
    let selected: Bool
    let running: Bool
    let select: () -> Void
    let play: () -> Void

    @State private var hovering = false
    @FocusState private var focused: Bool
    @ScaledMetric(relativeTo: .body) private var iconSize: CGFloat = 136

    var body: some View {
        VStack(spacing: 8) {
            artwork
            caption
            CompatBadge(status: game.compat?.status)
        }
        .frame(maxWidth: .infinity)
        .contentShape(.rect)
        .modifier(CardInteractions(game: game, selected: selected, select: select, play: play))
        .focusable()
        .focused($focused)
        .focusEffectDisabled()
        .onHover { hovering = $0 }
        .animation(.spring(duration: 0.3, bounce: 0.25), value: hovering)
        .animation(.spring(duration: 0.25), value: selected)
        .animation(.spring(duration: 0.35), value: running)
        .onChange(of: focused) { _, isFocused in
            if isFocused { select() }
        }
    }
}

private struct CardInteractions: ViewModifier {
    let game: GameEntry
    let selected: Bool
    let select: () -> Void
    let play: () -> Void

    private var traits: AccessibilityTraits { selected ? [.isButton, .isSelected] : [.isButton] }
    private var tooltip: String { game.error ?? game.compat?.note ?? game.path }

    func body(content: Content) -> some View {
        content
            .onTapGesture(count: 2, perform: play)
            .simultaneousGesture(TapGesture().onEnded(select))
            .onKeyPress(.return) {
                play()
                return .handled
            }
            .contextMenu { GameMenu(game: game, play: play) }
            .help(tooltip)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(game.accessibilityDescription)
            .accessibilityAddTraits(traits)
            .accessibilityAction(named: "Play") { play() }
            .accessibilityAction(.default) { select() }
    }
}

extension GameCard {
    private var radius: CGFloat { iconSize * 0.16 }
    private var traits: AccessibilityTraits { selected ? [.isButton, .isSelected] : [.isButton] }

    private var artwork: some View {
        GameIcon(game: game, size: iconSize)
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
            }
            .overlay(alignment: .topTrailing) { runningMark }
            .shadow(
                color: .black.opacity(hovering ? 0.28 : 0.18),
                radius: hovering ? 14 : 8, y: hovering ? 8 : 4
            )
            .padding(5)
            .overlay { selectionRing }
            .scaleEffect(hovering ? 1.03 : 1)
            .offset(y: hovering ? -3 : 0)
    }

    @ViewBuilder private var runningMark: some View {
        if running {
            Image(systemName: "play.fill")
                .font(.caption.weight(.bold))
                .padding(7)
                .glassEffect(.regular.tint(.accentColor), in: .circle)
                .padding(8)
                .transition(.scale.combined(with: .opacity))
        }
    }

    private var selectionRing: some View {
        RoundedRectangle(cornerRadius: radius + 5, style: .continuous)
            .strokeBorder(Color.accentColor, lineWidth: 3)
            .opacity(selected || focused ? 1 : 0)
    }

    private var caption: some View {
        VStack(spacing: 3) {
            Text(game.displayName)
                .font(.body.weight(.medium))
                .lineLimit(2)
                .multilineTextAlignment(.center)
            if let error = game.error {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
            } else {
                Text(game.titleId ?? "No title ID")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct GameTable: View {
    let games: [GameEntry]
    @Binding var selection: GameEntry.ID?
    let play: (GameEntry) -> Void
    @State private var order = [KeyPathComparator(\GameEntry.displayName)]

    var body: some View {
        Table(sorted, selection: $selection, sortOrder: $order) {
            TableColumn("") { GameIcon(game: $0, size: 30) }
                .width(34)
            TableColumn("Name", value: \.displayName)
            TableColumn("Title ID", value: \.titleIdText) { Text($0.titleIdText).monospaced() }
                .width(min: 80, ideal: 96)
            TableColumn("Version", value: \.versionText)
                .width(min: 60, ideal: 84)
            TableColumn("Status", value: \.statusRank) { CompatBadge(status: $0.compat?.status) }
                .width(min: 80, ideal: 92)
            TableColumn("Note") { game in
                Text(game.error ?? game.compat?.note ?? "")
                    .foregroundStyle(game.error == nil ? Color.secondary : Color.red)
                    .help(game.error ?? game.compat?.note ?? "")
            }
        }
        .contextMenu(forSelectionType: GameEntry.ID.self) { ids in
            if let game = games.first(where: { ids.contains($0.id) }) {
                GameMenu(game: game, play: { play(game) })
            }
        } primaryAction: { ids in
            if let game = games.first(where: { ids.contains($0.id) }) { play(game) }
        }
    }

    private var sorted: [GameEntry] { games.sorted(using: order) }
}

struct GameMenu: View {
    let game: GameEntry
    let play: () -> Void
    @Environment(Reporting.self) private var reporting

    var body: some View {
        Button("Play", systemImage: "play.fill", action: play)
            .disabled(game.error != nil)
        Button("Show in Finder", systemImage: "folder") {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: game.path)])
        }
        if let id = game.titleId {
            Button("Copy Title ID", systemImage: "doc.on.doc") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(id, forType: .string)
            }
            Button("Report Compatibility…", systemImage: "exclamationmark.bubble") { reporting.start(game) }
        }
    }
}

extension GameEntry {
    var titleIdText: String { titleId ?? "" }
    var versionText: String { version ?? "" }
    var statusRank: Int { compat?.status.rank ?? -2 }

    var accessibilityDescription: String {
        var parts = [displayName]
        if let titleId { parts.append(titleId) }
        if let version { parts.append("version \(version)") }
        parts.append(compat.map { "compatibility \($0.status.label)" } ?? "untested")
        if let error { parts.append(error) }
        return parts.joined(separator: ", ")
    }
}
