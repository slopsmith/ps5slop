import AppKit
import LauncherCore
import SwiftUI

@MainActor
@Observable
final class Updates {
    enum Prompt {
        case available(Nightly)
        case upToDate
        case localBuild
        case failed(String)
    }

    var prompt: Prompt?

    func checkOnLaunch() async {
        let enabled = UserDefaults.standard.object(forKey: Defaults.checkForUpdates) as? Bool ?? true
        guard enabled, LauncherApp.argument("--snapshot") == nil,
            UpdateCheck.isRelease(BuildInfo.current.commit),
            let nightly = try? await UpdateCheck.fetch(source: AboutView.source),
            case .available = UpdateCheck.status(current: BuildInfo.current.commit, nightly: nightly)
        else { return }
        prompt = .available(nightly)
    }

    func checkNow() async {
        let commit = BuildInfo.current.commit
        guard UpdateCheck.isRelease(commit) else {
            prompt = .localBuild
            return
        }
        do {
            switch UpdateCheck.status(current: commit, nightly: try await UpdateCheck.fetch(source: AboutView.source)) {
            case .available(let nightly): prompt = .available(nightly)
            case .upToDate: prompt = .upToDate
            case .localBuild: prompt = .localBuild
            }
        } catch {
            prompt = .failed(error.localizedDescription)
        }
    }
}

struct UpdateAlert: ViewModifier {
    @Bindable var updates: Updates

    private var isPresented: Binding<Bool> {
        Binding(get: { updates.prompt != nil }, set: { if !$0 { updates.prompt = nil } })
    }

    func body(content: Content) -> some View {
        content.alert(title, isPresented: isPresented, presenting: updates.prompt) { prompt in
            if case .available(let nightly) = prompt {
                Button("Download") { NSWorkspace.shared.open(nightly.download) }
                Button("Release Notes") { NSWorkspace.shared.open(nightly.release) }
                Button("Not Now", role: .cancel) {}
            } else {
                Button("OK", role: .cancel) {}
            }
        } message: { prompt in
            Text(message(prompt))
        }
    }

    private var title: String {
        switch updates.prompt {
        case .available: "A newer ps5slop is available"
        case .upToDate: "ps5slop is up to date"
        case .localBuild: "This is a local build"
        case .failed: "Could not check for updates"
        case nil: ""
        }
    }

    private func message(_ prompt: Updates.Prompt) -> String {
        let current = String((BuildInfo.current.commit ?? "").prefix(7))
        switch prompt {
        case .available(let nightly):
            return "The nightly from \(nightly.date.prefix(10)) (\(nightly.commit.prefix(7))) is out. You have \(current)."
        case .upToDate:
            return "You have the newest nightly (\(current))."
        case .localBuild:
            return "Only nightly builds from GitHub check for updates."
        case .failed(let reason):
            return reason
        }
    }
}
