import AppKit
import LauncherCore
import SwiftUI

@main
struct LauncherApp: App {
    @State private var library = Library(source: LauncherApp.librarySource())
    @State private var session = GameSession()

    var body: some Scene {
        Window("ps5slop", id: "library") {
            ContentView()
                .environment(library)
                .environment(session)
                .frame(minWidth: 760, minHeight: 480)
                .task { await Snapshot.takeIfAsked() }
        }
        .defaultSize(width: 1100, height: 720)
        .commands {
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

        Settings {
            SettingsView()
        }
    }

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
}
