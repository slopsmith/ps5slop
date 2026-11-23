import LauncherCore
import SwiftUI

struct GameInspector: View {
    let game: GameEntry
    let play: (GameEntry) -> Void
    @Environment(GameSession.self) private var session
    @Namespace private var glass

    private var isRunning: Bool { session.game?.id == game.id }
    private var showsLog: Bool { session.logGameID == game.id }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    actions
                    details
                    if let note = game.compat?.note {
                        section("Compatibility note") {
                            Text(note).textSelection(.enabled)
                        }
                    }
                    if let error = game.error {
                        section("Problem") {
                            Label(error, systemImage: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                        }
                    }
                }
                .padding(.horizontal, 18)
                .padding(.bottom, 18)
            }
            if showsLog {
                Divider()
                LogPane()
                    .frame(minHeight: 160, idealHeight: 240)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(duration: 0.35), value: showsLog)
    }

    private var header: some View {
        VStack(spacing: 14) {
            GameIcon(game: game, size: 168)
                .shadow(color: .black.opacity(0.25), radius: 16, y: 8)
                .padding(.top, 24)
            VStack(spacing: 6) {
                Text(game.displayName)
                    .font(.title2.weight(.semibold))
                    .multilineTextAlignment(.center)
                CompatBadge(status: game.compat?.status, showsSymbol: true)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.bottom, 4)
        .background(alignment: .top) {
            if let image = IconCache.image(game.icon) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(height: 190)
                    .blur(radius: 40)
                    .opacity(0.45)
                    .clipped()
                    .backgroundExtensionEffect()
                    .mask(LinearGradient(colors: [.black, .clear], startPoint: .top, endPoint: .bottom))
                    .accessibilityHidden(true)
            }
        }
    }

    private var actions: some View {
        GlassEffectContainer(spacing: 12) {
            HStack(spacing: 12) {
                if isRunning {
                    Button("Stop", systemImage: "stop.fill") { session.stop() }
                        .buttonStyle(.glass)
                        .controlSize(.extraLarge)
                        .frame(maxWidth: .infinity)
                        .glassEffectID("primary", in: glass)
                    if let started = session.startedAt {
                        Label {
                            Text(started, style: .timer).monospacedDigit()
                        } icon: {
                            Image(systemName: "clock")
                        }
                        .font(.callout)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .glassEffect(.regular, in: .capsule)
                        .glassEffectID("timer", in: glass)
                        .accessibilityLabel("Running time")
                    }
                } else {
                    Button("Play", systemImage: "play.fill") { play(game) }
                        .buttonStyle(.glassProminent)
                        .controlSize(.extraLarge)
                        .frame(maxWidth: .infinity)
                        .disabled(game.error != nil || session.isRunning)
                        .glassEffectID("primary", in: glass)
                }
                Button("Show in Finder", systemImage: "folder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: game.path)])
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.glass)
                .controlSize(.extraLarge)
                .help("Show in Finder")
                .glassEffectID("finder", in: glass)
            }
        }
        .animation(.spring(duration: 0.4, bounce: 0.2), value: isRunning)
    }

    private var details: some View {
        section("Details") {
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                row("Title ID", game.titleId, monospaced: true)
                row("Version", game.version, monospaced: true)
                row("Type", game.kind == .pkg ? "Package" : "App folder")
                row("Location", game.path)
            }
        }
    }

    private func row(_ label: String, _ value: String?, monospaced: Bool = false) -> some View {
        GridRow {
            Text(label)
                .foregroundStyle(.secondary)
                .gridColumnAlignment(.trailing)
            Text(value ?? "Unknown")
                .font(monospaced ? .body.monospaced() : .body)
                .foregroundStyle(value == nil ? .secondary : .primary)
                .lineLimit(3)
                .truncationMode(.middle)
                .textSelection(.enabled)
        }
    }

    private func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            content()
                .font(.callout)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct LogPane: View {
    @Environment(GameSession.self) private var session

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Output", systemImage: "text.alignleft")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Copy Output", systemImage: "doc.on.doc") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(session.log.lines.joined(separator: "\n"), forType: .string)
                }
                Button("Clear Output", systemImage: "trash") { session.clearLog() }
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        ForEach(Array(session.log.lines.enumerated()), id: \.offset) { item in
                            Text(item.element)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(item.offset)
                        }
                    }
                    .padding(10)
                }
                .onChange(of: session.log.lines.count) { _, count in
                    proxy.scrollTo(count - 1, anchor: .bottom)
                }
            }
            .background(.background.secondary)
        }
    }
}

struct NowPlayingBar: View {
    let game: GameEntry
    let startedAt: Date?
    let stop: () -> Void
    let show: () -> Void

    var body: some View {
        GlassEffectContainer(spacing: 8) {
            HStack(spacing: 12) {
                Button(action: show) {
                    HStack(spacing: 10) {
                        GameIcon(game: game, size: 34)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(game.displayName)
                                .font(.callout.weight(.semibold))
                                .lineLimit(1)
                            Group {
                                if let startedAt {
                                    Text("Playing for \(Text(startedAt, style: .timer))")
                                } else {
                                    Text("Starting")
                                }
                            }
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.leading, 6)
                    .contentShape(.rect)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Now playing \(game.displayName)")
                .accessibilityHint("Shows the game in the library")

                Button("Stop", systemImage: "stop.fill", action: stop)
                    .labelStyle(.iconOnly)
                    .buttonStyle(.glass)
                    .controlSize(.large)
                    .help("Quit the running game")
            }
            .padding(6)
            .padding(.trailing, 2)
            .glassEffect(.regular.interactive(), in: .capsule)
        }
    }
}
