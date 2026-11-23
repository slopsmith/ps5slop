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
    @AppStorage(Defaults.showLog) private var showLog = false
    @State private var sidebar: SidebarItem? = .all
    @State private var selection: GameEntry.ID?
    @State private var search = ""
    @State private var dropTargeted = false

    var body: some View {
        NavigationSplitView {
            Sidebar(selection: $sidebar)
                .navigationSplitViewColumnWidth(min: 180, ideal: 210)
        } detail: {
            VSplitView {
                libraryView
                    .frame(maxWidth: .infinity, minHeight: 200, maxHeight: .infinity)
                if showLog {
                    LogPane()
                        .frame(maxWidth: .infinity, minHeight: 90, idealHeight: 160, maxHeight: 400)
                }
            }
            .navigationTitle(title)
            .navigationSubtitle(subtitle)
        }
        .searchable(text: $search, placement: .toolbar, prompt: "Name or title ID")
        .toolbar { toolbar }
        .dropDestination(for: URL.self) { urls, _ in
            let usable = urls.filter(Self.isUsable)
            library.add(usable)
            return !usable.isEmpty
        } isTargeted: { dropTargeted = $0 }
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color.accentColor, lineWidth: 3)
                    .padding(4)
                    .allowsHitTesting(false)
            }
        }
        .alert(
            "Cannot start the game",
            isPresented: Binding(get: { session.lastError != nil }, set: { _ in }),
            presenting: session.lastError
        ) { _ in
            Button("OK") {}
        } message: { Text($0) }
    }

    @ViewBuilder private var libraryView: some View {
        let games = visibleGames
        if library.folders.isEmpty {
            ContentUnavailableView {
                Label("No Games", systemImage: "gamecontroller")
            } description: {
                Text("Add a folder with your games, or drop one here.")
            } actions: {
                Button("Add Games Folder…") { AddFolderPanel.run(library) }
            }
        } else if games.isEmpty {
            if library.isScanning {
                ProgressView("Scanning…")
            } else if !search.isEmpty {
                ContentUnavailableView.search(text: search)
            } else {
                ContentUnavailableView(
                    "No Games Found", systemImage: "questionmark.folder",
                    description: Text(library.scanError ?? "No .pkg files or app folders in these folders."))
            }
        } else {
            VStack(spacing: 0) {
                if let error = library.scanError {
                    ErrorBanner(text: error)
                }
                switch viewMode {
                case .grid: GameGrid(games: games, selection: $selection, play: play)
                case .list: GameTable(games: games, selection: $selection, play: play)
                }
            }
        }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            if session.isRunning {
                Button("Stop", systemImage: "stop.fill") { session.stop() }
                    .help("Quit the running game")
            } else {
                Button("Play", systemImage: "play.fill") {
                    if let game = selectedGame { play(game) }
                }
                .disabled(selectedGame == nil)
                .help("Start the selected game")
            }
            Picker("View", selection: $viewMode) {
                Label("Icons", systemImage: "square.grid.2x2").tag(ViewMode.grid)
                Label("List", systemImage: "list.bullet").tag(ViewMode.list)
            }
            .pickerStyle(.segmented)
            .help("Show games as icons or as a list")
            Button("Refresh", systemImage: "arrow.clockwise") { library.refresh() }
                .disabled(library.isScanning)
                .help("Scan the game folders again")
            Toggle("Log", systemImage: "text.alignleft", isOn: $showLog)
                .help("Show the running game's output")
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
        if let game = session.game { return "Running \(game.displayName)" }
        if library.isScanning { return "Scanning…" }
        let n = visibleGames.count
        return n == 1 ? "1 game" : "\(n) games"
    }

    private func play(_ game: GameEntry) {
        guard !session.isRunning, game.error == nil else { return }
        session.launch(game, dataFolder: dataFolder, fontsFolder: fontsFolder)
        showLog = true
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
            Label("All Games", systemImage: "square.grid.2x2").tag(SidebarItem.all)
            Section("Folders") {
                ForEach(library.folders, id: \.self) { folder in
                    Label(folder.lastPathComponent, systemImage: folder.pathExtension == "pkg" ? "shippingbox" : "folder")
                        .help(folder.path)
                        .tag(SidebarItem.folder(folder))
                        .contextMenu {
                            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([folder]) }
                            Button("Remove from Library", role: .destructive) {
                                if selection == .folder(folder) { selection = .all }
                                library.remove(folder)
                            }
                        }
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            HStack {
                Button("Add Games Folder…", systemImage: "plus") { AddFolderPanel.run(library) }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .help("Add a folder with games")
                if library.isScanning { ProgressView().controlSize(.small) }
                Spacer()
            }
            .padding(8)
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
            .padding(8)
            .background(.yellow.opacity(0.15))
    }
}
