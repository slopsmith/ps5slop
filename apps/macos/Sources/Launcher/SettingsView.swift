import AppKit
import LauncherCore
import SwiftUI

struct SettingsView: View {
    @AppStorage(Defaults.dataFolder) private var dataFolder = ""
    @AppStorage(Defaults.fontsFolder) private var fontsFolder = ""
    @AppStorage(Defaults.checkForUpdates) private var checkForUpdates = true
    @State private var fonts = FontFetcher()

    var body: some View {
        Form {
            Section {
                FolderField(title: "Data folder", path: $dataFolder, placeholder: "One per title in Application Support")
            } footer: {
                Text("The folder games see as /data, with their saves next to it.")
            }
            Section {
                FolderField(title: "Fonts folder", path: $fontsFolder, placeholder: Paths.defaultFontsFolder.path)
                HStack {
                    Button("Download Fonts") { fonts.fetch(into: fontsFolder.isEmpty ? Paths.defaultFontsFolder : URL(fileURLWithPath: fontsFolder)) }
                        .disabled(fonts.isRunning)
                    if fonts.isRunning { ProgressView().controlSize(.small) }
                    Text(fonts.status).foregroundStyle(.secondary).lineLimit(2)
                }
            } footer: {
                Text("Open-source stand-ins for the console's system fonts, fetched from Google Fonts. Some games, such as Astro Bot, need them.")
            }
            Section {
                Toggle("Check for updates when ps5slop starts", isOn: $checkForUpdates)
            } footer: {
                Text("Compares this build with the newest nightly on GitHub. Local builds never check.")
            }
            Section {
                Button("Open Captures Folder") { openCaptures() }
            } footer: {
                Text("F9 in a game saves a GPU frame capture here.")
            }
        }
        .formStyle(.grouped)
        .frame(width: 520)
        .fixedSize(horizontal: false, vertical: true)
    }
}

struct FolderField: View {
    let title: String
    @Binding var path: String
    let placeholder: String

    var body: some View {
        LabeledContent(title) {
            HStack {
                Text(path.isEmpty ? placeholder : path)
                    .foregroundStyle(path.isEmpty ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(path.isEmpty ? placeholder : path)
                Button("Choose…") { choose() }
                if !path.isEmpty {
                    Button("Reset", systemImage: "xmark.circle.fill") { path = "" }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .help("Use the default")
                }
            }
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}

@MainActor
@Observable
final class FontFetcher {
    private(set) var isRunning = false
    private(set) var status = ""

    func fetch(into folder: URL) {
        guard let script = BinaryLocator.fetchFontsScript() else {
            status = "scripts/fetch-fonts was not found."
            return
        }
        isRunning = true
        status = "Downloading…"
        Task {
            let ok = await Self.run(script: script, folder: folder)
            isRunning = false
            status = ok ? "Fonts are in \(folder.path)." : "Download failed."
        }
    }

    private nonisolated static func run(script: URL, folder: URL) async -> Bool {
        await withCheckedContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = [script.path, folder.path]
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            process.terminationHandler = { continuation.resume(returning: $0.terminationStatus == 0) }
            do {
                try process.run()
            } catch {
                process.terminationHandler = nil
                continuation.resume(returning: false)
            }
        }
    }
}
