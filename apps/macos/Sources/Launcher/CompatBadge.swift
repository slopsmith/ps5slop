import LauncherCore
import SwiftUI

struct CompatBadge: View {
    let status: CompatStatus?

    var body: some View {
        Text(status?.label ?? "Untested")
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .foregroundStyle(status == nil ? Color.secondary : Color.white)
            .background(color.opacity(status == nil ? 0.2 : 1), in: Capsule())
            .accessibilityLabel("Compatibility: \(status?.label ?? "untested")")
    }

    private var color: Color {
        switch status {
        case .perfect: .purple
        case .playable: .green
        case .ingame: .yellow.mix(with: .orange, by: 0.3)
        case .menus: .orange
        case .boots: .red
        case .nothing: .gray
        case .other, nil: .secondary
        }
    }
}

struct GameIcon: View {
    let game: GameEntry
    var size: CGFloat

    var body: some View {
        Group {
            if let image = IconCache.image(game.icon) {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fill)
            } else {
                ZStack {
                    Rectangle().fill(.quaternary)
                    Image(systemName: game.error == nil ? "gamecontroller" : "exclamationmark.triangle")
                        .font(.system(size: size * 0.35))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.12, style: .continuous))
    }
}

@MainActor
enum IconCache {
    private static let cache = NSCache<NSString, NSImage>()

    static func image(_ path: String?) -> NSImage? {
        guard let path else { return nil }
        if let hit = cache.object(forKey: path as NSString) { return hit }
        guard let image = NSImage(contentsOfFile: path) else { return nil }
        cache.setObject(image, forKey: path as NSString)
        return image
    }
}
