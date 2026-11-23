import SwiftUI

struct LogPane: View {
    @Environment(GameSession.self) private var session

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(session.game.map { "Output: \($0.displayName)" } ?? "Output")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Copy", systemImage: "doc.on.doc") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(session.log.lines.joined(separator: "\n"), forType: .string)
                }
                Button("Clear", systemImage: "trash") { session.clearLog() }
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(session.log.lines.enumerated()), id: \.offset) { item in
                            Text(item.element)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(item.offset)
                        }
                    }
                    .padding(8)
                }
                .onChange(of: session.log.lines.count) { _, count in
                    proxy.scrollTo(count - 1, anchor: .bottom)
                }
            }
            .background(Color(nsColor: .textBackgroundColor))
        }
    }
}
