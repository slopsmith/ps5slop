import LauncherCore
import SwiftUI

extension CompatStatus {
    var tint: Color {
        switch self {
        case .perfect: .mint
        case .playable: .green
        case .ingame: .yellow
        case .menus: .orange
        case .boots: .red
        case .nothing, .other: .gray
        }
    }

    var meaning: String {
        switch self {
        case .perfect: "Plays through with no known problems."
        case .playable: "Can be played start to end; may have glitches or slowdowns."
        case .ingame: "Gets into gameplay, but problems keep it from being played properly."
        case .menus: "Reaches the title screen or menus, not gameplay."
        case .boots: "Starts and shows something (a logo, a loading screen), no menus."
        case .nothing: "Stops or crashes before showing anything."
        case .other: ""
        }
    }

    var symbol: String {
        switch self {
        case .perfect: "star.fill"
        case .playable: "checkmark.circle.fill"
        case .ingame: "gamecontroller.fill"
        case .menus: "list.bullet.rectangle.fill"
        case .boots: "power.circle.fill"
        case .nothing, .other: "xmark.circle.fill"
        }
    }
}

struct CompatBadge: View {
    let status: CompatStatus?
    var showsSymbol = false

    var body: some View {
        Label {
            Text(status?.label ?? "Untested")
        } icon: {
            if showsSymbol { Image(systemName: status?.symbol ?? "questionmark.circle") }
        }
        .labelStyle(BadgeLabelStyle(showsSymbol: showsSymbol))
        .font(.caption.weight(.semibold))
        .foregroundStyle(foreground)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(background, in: .capsule)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Compatibility: \(status?.label ?? "untested")")
    }

    private var foreground: AnyShapeStyle {
        guard let status else { return AnyShapeStyle(.secondary) }
        return status == .ingame ? AnyShapeStyle(Color.black.opacity(0.8)) : AnyShapeStyle(.white)
    }

    private var background: AnyShapeStyle {
        guard let status else { return AnyShapeStyle(.quaternary) }
        return AnyShapeStyle(status.tint.gradient)
    }
}

private struct BadgeLabelStyle: LabelStyle {
    let showsSymbol: Bool

    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) {
            if showsSymbol { configuration.icon }
            configuration.title
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
                    Rectangle().fill(.fill.tertiary)
                    Image(systemName: game.error == nil ? "gamecontroller" : "exclamationmark.triangle")
                        .font(.system(size: size * 0.32, weight: .light))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(.rect(cornerRadius: size * 0.16, style: .continuous))
        .accessibilityHidden(true)
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
