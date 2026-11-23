import LauncherCore
import SwiftUI

struct GameGrid: View {
    let games: [GameEntry]
    @Binding var selection: GameEntry.ID?
    let play: (GameEntry) -> Void

    var body: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150, maximum: 190), spacing: 16)], spacing: 20) {
                ForEach(games) { game in
                    GameTile(game: game, selected: selection == game.id)
                        .onTapGesture(count: 2) { play(game) }
                        .simultaneousGesture(TapGesture().onEnded { selection = game.id })
                        .contextMenu { GameMenu(game: game, play: play) }
                }
            }
            .padding(20)
        }
        .background(Color(nsColor: .textBackgroundColor).opacity(0.001))
        .onTapGesture { selection = nil }
    }
}

struct GameTile: View {
    let game: GameEntry
    let selected: Bool

    var body: some View {
        VStack(spacing: 6) {
            GameIcon(game: game, size: 128)
                .padding(4)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(selected ? Color.accentColor.opacity(0.25) : .clear))
            Text(game.displayName)
                .font(.callout.weight(.medium))
                .lineLimit(2)
                .multilineTextAlignment(.center)
            if let error = game.error {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
            } else {
                Text([game.titleId, game.version].compactMap(\.self).joined(separator: " · "))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            CompatBadge(status: game.compat?.status)
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .help(game.error ?? game.compat?.note ?? game.path)
    }
}

struct GameTable: View {
    let games: [GameEntry]
    @Binding var selection: GameEntry.ID?
    let play: (GameEntry) -> Void
    @State private var order = [KeyPathComparator(\GameEntry.displayName)]

    var body: some View {
        Table(sorted, selection: $selection, sortOrder: $order) {
            TableColumn("") { GameIcon(game: $0, size: 28) }
                .width(32)
            TableColumn("Name", value: \.displayName)
            TableColumn("Title ID", value: \.titleIdText) { Text($0.titleIdText).monospaced() }
                .width(min: 80, ideal: 90)
            TableColumn("Version", value: \.versionText)
                .width(min: 60, ideal: 70)
            TableColumn("Status", value: \.statusRank) { CompatBadge(status: $0.compat?.status) }
                .width(min: 70, ideal: 80)
            TableColumn("Note") { game in
                Text(game.error ?? game.compat?.note ?? "")
                    .foregroundStyle(game.error == nil ? Color.secondary : Color.red)
                    .help(game.error ?? game.compat?.note ?? "")
            }
        }
        .contextMenu(forSelectionType: GameEntry.ID.self) { ids in
            if let game = games.first(where: { ids.contains($0.id) }) {
                GameMenu(game: game, play: play)
            }
        } primaryAction: { ids in
            if let game = games.first(where: { ids.contains($0.id) }) { play(game) }
        }
    }

    private var sorted: [GameEntry] { games.sorted(using: order) }
}

struct GameMenu: View {
    let game: GameEntry
    let play: (GameEntry) -> Void

    var body: some View {
        Button("Play") { play(game) }
            .disabled(game.error != nil)
        Button("Show in Finder") {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: game.path)])
        }
        if let id = game.titleId {
            Button("Copy Title ID") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(id, forType: .string)
            }
        }
    }
}

extension GameEntry {
    var titleIdText: String { titleId ?? "" }
    var versionText: String { version ?? "" }
    var statusRank: Int { compat?.status.rank ?? -2 }
}
