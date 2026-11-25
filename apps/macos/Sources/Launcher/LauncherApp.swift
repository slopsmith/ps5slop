import AppKit
import LauncherCore
import SwiftUI

@main
struct LauncherApp: App {
    @State private var library = Library(source: LauncherApp.librarySource())
    @State private var session = GameSession()
    @State private var updates = Updates()
    @State private var reporting = Reporting()

    var body: some Scene {
        Window("ps5slop", id: "library") {
            if Self.showsAbout {
                AboutView()
                    .task { await Snapshot.takeIfAsked() }
            } else {
                ContentView()
                    .environment(library)
                    .environment(session)
                    .environment(reporting)
                    .frame(minWidth: 760, minHeight: 480)
                    .task {
                        await Snapshot.takeIfAsked {
                            let id = Self.argument("--select")
                            guard let game = library.games.first(where: { $0.titleId == id }) else { return }
                            let answers = (Self.argument("--report-answers") ?? "").split(separator: ",")
                            reporting.start(
                                game,
                                answers: Dictionary(
                                    uniqueKeysWithValues: zip(CompatReport.Question.allCases, answers.map { $0 == "yes" })))
                        }
                    }
                    .task { await updates.checkOnLaunch() }
                    .task { await library.refreshCompat() }
                    .modifier(UpdateAlert(updates: updates))
            }
        }
        .defaultSize(width: Self.showsAbout ? 860 : 1100, height: Self.showsAbout ? 600 : 720)
        .commands {
            CommandGroup(replacing: .appInfo) {
                AboutButton()
                Button("Check for Updates…") { Task { await updates.checkNow() } }
            }
            CommandGroup(after: .newItem) {
                Button("Add Games Folder…") { AddFolderPanel.run(library) }
                    .keyboardShortcut("o")
                Button("Refresh Library") { library.refresh() }
                    .keyboardShortcut("r")
            }
            CommandGroup(after: .sidebar) {
                Button("Open Captures Folder") { openCaptures() }
            }
        }

        Window("About ps5slop", id: "about") {
            AboutView()
        }
        .defaultSize(width: 860, height: 600)
        .defaultLaunchBehavior(.suppressed)
        .windowResizability(.contentMinSize)
        .restorationBehavior(.disabled)

        Settings {
            SettingsView()
        }
    }

    /// `--about` shows the About view in the main window instead of the library, for `--snapshot`.
    static let showsAbout = CommandLine.arguments.contains("--about")

    static func argument(_ name: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    /// `--fixture PATH` shows a saved `info` JSON instead of the real library.
    private static func librarySource() -> Library.Source {
        if let path = argument("--fixture") {
            return .fixture(URL(fileURLWithPath: path))
        }
        return .store(.standard)
    }
}

struct AboutButton: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("About ps5slop") { openWindow(id: "about") }
    }
}

@MainActor
func openCaptures() {
    let folder = Paths.capturesFolder
    try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    NSWorkspace.shared.open(folder)
}

@MainActor
enum AddFolderPanel {
    static func run(_ library: Library) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.folder, .init(filenameExtension: "pkg") ?? .data]
        panel.allowsMultipleSelection = true
        panel.prompt = "Add"
        panel.message = "Choose folders with games, app folders or .pkg files."
        if panel.runModal() == .OK {
            library.add(panel.urls)
        }
    }
}

enum Defaults {
    static let dataFolder = "dataFolder"
    static let fontsFolder = "fontsFolder"
    static let viewMode = "viewMode"
    static let showInspector = "showInspector"
    static let checkForUpdates = "checkForUpdates"
}
