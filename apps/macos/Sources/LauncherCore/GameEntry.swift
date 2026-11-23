import Foundation

public enum CompatStatus: Sendable, Hashable, Codable, Comparable {
    case nothing, boots, menus, ingame, playable, perfect
    case other(String)

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self.known[raw.lowercased()] ?? .other(raw)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(label)
    }

    private static let known: [String: CompatStatus] = [
        "nothing": .nothing, "boots": .boots, "menus": .menus,
        "ingame": .ingame, "playable": .playable, "perfect": .perfect,
    ]

    public var label: String {
        switch self {
        case .nothing: "Nothing"
        case .boots: "Boots"
        case .menus: "Menus"
        case .ingame: "Ingame"
        case .playable: "Playable"
        case .perfect: "Perfect"
        case .other(let raw): raw
        }
    }

    public var rank: Int {
        switch self {
        case .other: -1
        case .nothing: 0
        case .boots: 1
        case .menus: 2
        case .ingame: 3
        case .playable: 4
        case .perfect: 5
        }
    }

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rank < rhs.rank }
}

public struct Compat: Sendable, Hashable, Codable {
    public var status: CompatStatus
    public var note: String?

    public init(status: CompatStatus, note: String? = nil) {
        self.status = status
        self.note = note
    }
}

/// One game as reported by `ps5slop info --json`.
public struct GameEntry: Sendable, Hashable, Codable, Identifiable {
    public enum Kind: String, Sendable, Hashable, Codable {
        case pkg, folder, unknown

        public init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Kind(rawValue: raw) ?? .unknown
        }
    }

    public var path: String
    public var kind: Kind
    public var titleId: String?
    public var name: String?
    public var version: String?
    public var icon: String?
    public var compat: Compat?
    public var error: String?

    public var id: String { path }

    public var displayName: String {
        name ?? URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
    }

    public init(
        path: String, kind: Kind, titleId: String? = nil, name: String? = nil,
        version: String? = nil, icon: String? = nil, compat: Compat? = nil, error: String? = nil
    ) {
        self.path = path
        self.kind = kind
        self.titleId = titleId
        self.name = name
        self.version = version
        self.icon = icon
        self.compat = compat
        self.error = error
    }

    public func matches(_ query: String) -> Bool {
        let q = query.trimmingCharacters(in: .whitespaces)
        if q.isEmpty { return true }
        return displayName.localizedCaseInsensitiveContains(q)
            || (titleId?.localizedCaseInsensitiveContains(q) ?? false)
    }

    public func isInside(_ folder: URL) -> Bool {
        let base = folder.standardizedFileURL.path
        let own = URL(fileURLWithPath: path).standardizedFileURL.path
        return own == base || own.hasPrefix(base.hasSuffix("/") ? base : base + "/")
    }
}

public enum GameInfoDecoder {
    public static func decode(_ data: Data) throws -> [GameEntry] {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([GameEntry].self, from: data)
    }

    public static func encode(_ games: [GameEntry]) throws -> Data {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(games)
    }
}
