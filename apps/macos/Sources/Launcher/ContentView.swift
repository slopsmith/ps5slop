import LauncherCore
import SwiftUI

enum SidebarItem: Hashable {
    case all
    case folder(URL)
}

enum ViewMode: String {
    case grid, list
}

struct ContentView: View {
    @Environment(Library.self) private var library
    @Environment(GameSession.self) private var session
    @AppStorage(Defaults.dataFolder) private var dataFolder = ""
    @AppStorage(Defaults.fontsFolder) private var fontsFolder = ""
    @AppStorage(Defaults.viewMode) private var viewMode = ViewMode.grid
    @AppStorage(Defaults.showInspector) private var showInspector = true
    @State private var sidebar: SidebarItem? = .all
    @State private var selection: GameEntry.ID?
    @State private var search = ""
    @State private var dropTargeted = false
    @Environment(Reporting.self) private var reporting
    @Namespace private var glass

    var body: some View {
        NavigationSplitView {
            Sidebar(selection: $sidebar)
                .navigationSplitViewColumnWidth(min: 180, ideal: 220)
        } detail: {
            libraryView
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay(alignment: .bottom) { nowPlaying }
                .navigationTitle(title)
                .navigationSubtitle(subtitle)
                .inspector(isPresented: $showInspector) {
                    inspector
                        .inspectorColumnWidth(min: 280, ideal: 320, max: 420)
                }
        }
        .searchable(text: $search, placement: .toolbar, prompt: "Name or title ID")
        .toolbar { toolbar }
        .dropDestination(for: URL.self) { urls, _ in
            let usable = urls.filter(Self.isUsable)
            library.add(usable)
            return !usable.isEmpty
        } isTargeted: { dropTargeted = $0 }
        .overlay { dropHighlight }
        .animation(.spring(duration: 0.3), value: dropTargeted)
        .animation(.spring(duration: 0.45, bounce: 0.2), value: session.game?.id)
        .alert(
            "Cannot start the game",
            isPresented: Binding(get: { session.lastError != nil }, set: { _ in }),
            presenting: session.lastError
        ) { _ in
            Button("OK") {}
        } message: { Text($0) }
        .sheet(item: Bindable(reporting).game) { game in
            ReportSheet(game: game, answers: reporting.answers)
        }
        .onAppear {
            if let id = LauncherApp.argument("--select") {
                selection = library.games.first { $0.titleId == id }?.id
            }
        }
    }

    @ViewBuilder private var libraryView: some View {
        let games = visibleGames
        if library.folders.isEmpty {
            ContentUnavailableView {
                Label("Your Library Is Empty", systemImage: "square.stack.3d.up")
            } description: {
                Text("Add a folder with your games, or drop folders and .pkg files here.")
            } actions: {
                Button("Add Folder…", systemImage: "folder.badge.plus") { AddFolderPanel.run(library) }
                    .buttonStyle(.glassProminent)
                    .controlSize(.large)
            }
        } else if games.isEmpty {
            if library.isScanning {
                ProgressView("Scanning…")
            } else if !search.isEmpty {
                ContentUnavailableView.search(text: search)
            } else {
                ContentUnavailableView {
                    Label("No Games Found", systemImage: "questionmark.folder")
                } description: {
                    Text(library.scanError ?? "No .pkg files or app folders in these folders.")
                } actions: {
                    Button("Scan Again", systemImage: "arrow.clockwise") { library.refresh() }
                        .buttonStyle(.glass)
                }
            }
        } else {
            VStack(spacing: 0) {
                if let error = library.scanError {
                    ErrorBanner(text: error)
                }
                switch viewMode {
                case .grid:
                    GameGrid(games: games, selection: $selection, runningID: session.game?.id, play: play)
                case .list:
                    GameTable(games: games, selection: $selection, play: play)
                }
            }
        }
    }

    @ViewBuilder private var inspector: some View {
        if let game = selectedGame {
            GameInspector(game: game, play: play)
                .id(game.id)
        } else {
            ContentUnavailableView(
                "No Game Selected", systemImage: "gamecontroller",
                description: Text("Select a game to see its details."))
        }
    }

    @ViewBuilder private var nowPlaying: some View {
        if let game = session.game {
            NowPlayingBar(
                game: game, startedAt: session.startedAt,
                stop: { session.stop() },
                show: {
                    selection = game.id
                    showInspector = true
                }
            )
            .glassEffectID("nowPlaying", in: glass)
            .padding(.bottom, 16)
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    @ViewBuilder private var dropHighlight: some View {
        if dropTargeted {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [10, 6]))
                .background(Color.accentColor.opacity(0.08), in: .rect(cornerRadius: 16, style: .continuous))
                .overlay {
                    Label("Add to Library", systemImage: "plus.circle.fill")
                        .font(.title3.weight(.semibold))
                        .padding(.horizontal, 18)
                        .padding(.vertical, 10)
                        .glassEffect(.regular.tint(.accentColor.opacity(0.4)), in: .capsule)
                }
                .padding(8)
                .allowsHitTesting(false)
                .transition(.opacity)
        }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Picker("View", selection: $viewMode) {
                Label("Icons", systemImage: "square.grid.2x2").tag(ViewMode.grid)
                Label("List", systemImage: "list.bullet").tag(ViewMode.list)
            }
            .pickerStyle(.segmented)
            .help("Show games as icons or as a list")
        }
        ToolbarItem(placement: .primaryAction) {
            Button("Refresh", systemImage: "arrow.clockwise") { library.refresh() }
                .disabled(library.isScanning)
                .help("Scan the game folders again")
        }
        ToolbarSpacer(.fixed, placement: .primaryAction)
        ToolbarItem(placement: .primaryAction) {
            if session.isRunning {
                Button("Stop", systemImage: "stop.fill") { session.stop() }
                    .help("Quit the running game")
            } else {
                Button("Play", systemImage: "play.fill") {
                    if let game = selectedGame { play(game) }
                }
                .buttonStyle(.glassProminent)
                .disabled(selectedGame == nil || selectedGame?.error != nil)
                .help("Start the selected game")
            }
        }
        ToolbarItem(placement: .primaryAction) {
            Button("Inspector", systemImage: "sidebar.trailing") { showInspector.toggle() }
                .help(showInspector ? "Hide the game details" : "Show the game details")
        }
    }

    private var selectedGame: GameEntry? {
        library.games.first { $0.id == selection }
    }

    private var visibleGames: [GameEntry] {
        library.games
            .filter { game in
                if case .folder(let url) = sidebar, !game.isInside(url) { return false }
                return game.matches(search)
            }
            .sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }

    private var title: String {
        if case .folder(let url) = sidebar { return url.lastPathComponent }
        return "All Games"
    }

    private var subtitle: String {
        if library.isScanning { return "Scanning…" }
        let n = visibleGames.count
        return n == 1 ? "1 game" : "\(n) games"
    }

    private func play(_ game: GameEntry) {
        guard !session.isRunning, game.error == nil else { return }
        selection = game.id
        session.launch(game, dataFolder: dataFolder, fontsFolder: fontsFolder)
    }

    private static func isUsable(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return false }
        return isDir.boolValue || url.pathExtension.lowercased() == "pkg"
    }
}

struct Sidebar: View {
    @Environment(Library.self) private var library
    @Binding var selection: SidebarItem?

    var body: some View {
        List(selection: $selection) {
            Section("Library") {
                Label("All Games", systemImage: "square.grid.2x2")
                    .badge(library.games.count)
                    .tag(SidebarItem.all)
            }
            Section("Folders") {
                ForEach(library.folders, id: \.self) { folder in
                    Label(folder.lastPathComponent, systemImage: folder.pathExtension == "pkg" ? "shippingbox" : "folder")
                        .badge(library.games.filter { $0.isInside(folder) }.count)
                        .help(folder.path)
                        .tag(SidebarItem.folder(folder))
                        .contextMenu {
                            Button("Show in Finder", systemImage: "folder") {
                                NSWorkspace.shared.activateFileViewerSelecting([folder])
                            }
                            Button("Remove from Library", systemImage: "minus.circle", role: .destructive) {
                                if selection == .folder(folder) { selection = .all }
                                library.remove(folder)
                            }
                        }
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            HStack {
                Button("Add Folder…", systemImage: "plus") { AddFolderPanel.run(library) }
                    .buttonStyle(.glass)
                    .help("Add a folder with games")
                if library.isScanning {
                    ProgressView().controlSize(.small)
                }
                Spacer()
            }
            .padding(10)
        }
    }
}

struct ErrorBanner: View {
    let text: String

    var body: some View {
        Label(text, systemImage: "exclamationmark.triangle.fill")
            .font(.callout)
            .lineLimit(2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .glassEffect(.regular.tint(.yellow.opacity(0.3)), in: .rect(cornerRadius: 12))
            .padding(.horizontal, 16)
            .padding(.top, 8)
    }
}
