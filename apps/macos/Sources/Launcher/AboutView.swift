import AppKit
import SwiftUI

struct AboutView: View {
    static let source = URL(string: "https://github.com/slopsmith/ps5slop")!

    private let info = BuildInfo.current
    private let licenses = LicenseFile.bundled()
    @State private var selection: LicenseFile.ID?

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(20)
            Divider()
            if licenses.isEmpty {
                ContentUnavailableView(
                    "No Licenses", systemImage: "doc.text",
                    description: Text("They are added when scripts/bundle-app builds the app."))
            } else {
                HStack(spacing: 0) {
                    List(licenses, selection: $selection) { file in
                        Text(file.title)
                    }
                    .listStyle(.sidebar)
                    .frame(width: 200)
                    Divider()
                    ScrollView {
                        Text(licenses.first { $0.id == selection }?.text ?? "")
                            .font(.system(.callout, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                    }
                    .frame(minWidth: 620)
                }
            }
        }
        .frame(minWidth: 860, minHeight: 560)
        .onAppear { selection = selection ?? licenses.first?.id }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 16) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 72, height: 72)
            VStack(alignment: .leading, spacing: 4) {
                Text(info.name).font(.title2.bold())
                Text("Version \(info.version)").foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    if let commit = info.commit {
                        Text("Commit")
                            .foregroundStyle(.secondary)
                        Link(String(commit.prefix(12)), destination: info.commitURL(source: Self.source))
                            .font(.body.monospaced())
                            .help(commit)
                        Button("Copy", systemImage: "doc.on.doc") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(commit, forType: .string)
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .help("Copy the full commit")
                    } else {
                        Text("Development build").foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            Link("Source Code", destination: Self.source)
        }
    }
}

struct BuildInfo {
    var name: String
    var version: String
    var commit: String?

    static var current: BuildInfo {
        let info = Bundle.main.infoDictionary ?? [:]
        return BuildInfo(
            name: info["CFBundleName"] as? String ?? ProcessInfo.processInfo.processName,
            version: info["CFBundleShortVersionString"] as? String ?? "unknown",
            commit: info["SlopCommit"] as? String)
    }

    var shortCommit: String {
        guard let commit else { return "unknown" }
        let parts = commit.split(separator: "-", maxSplits: 1)
        return parts.first.map { String($0.prefix(7)) + (parts.count > 1 ? "-\(parts[1])" : "") } ?? commit
    }

    func commitURL(source: URL) -> URL {
        let sha = commit?.split(separator: "-").first.map(String.init) ?? ""
        return source.appending(path: "commit/\(sha)")
    }
}

struct LicenseFile: Identifiable {
    var id: String { title }
    var title: String
    var text: String

    static func bundled() -> [LicenseFile] {
        guard let folder = Bundle.main.resourceURL?.appending(path: "Licenses"),
            let names = try? FileManager.default.contentsOfDirectory(atPath: folder.path)
        else { return [] }
        let own = "\(BuildInfo.current.name) ("
        func rank(_ name: String) -> Int { name == "README.txt" ? 0 : name.hasPrefix(own) ? 1 : 2 }
        return names.filter { $0.hasSuffix(".txt") }
            .sorted { rank($0) != rank($1) ? rank($0) < rank($1) : $0.localizedStandardCompare($1) == .orderedAscending }
            .compactMap { name in
                guard let text = try? String(contentsOf: folder.appending(path: name), encoding: .utf8) else { return nil }
                let title = name == "README.txt" ? "About These Licenses" : String(name.dropLast(4))
                return LicenseFile(title: title, text: text)
            }
    }
}
