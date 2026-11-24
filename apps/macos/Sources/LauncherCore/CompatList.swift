import Foundation

/// The compatibility list the website builds from the reports in slopsmith/ps5slop-compat.
public struct CompatList: Decodable, Sendable, Equatable {
    public struct Report: Decodable, Sendable, Equatable {
        public var url: String
        public var date: String
        public var status: CompatStatus
        public var notes: String?
        public var macos: String
        public var chip: String
        public var ps5slop: String
    }

    public struct Title: Decodable, Sendable, Equatable {
        public var name: String
        public var status: CompatStatus
        public var reports: [Report]
    }

    public var version: Int
    public var titles: [String: Title]

    public static let url = URL(string: "https://slopsmith.github.io/ps5slop/compat.json")!

    public static func decode(_ data: Data) throws -> CompatList {
        let list = try JSONDecoder().decode(CompatList.self, from: data)
        guard list.version == 1 else { throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "compat.json version \(list.version)")) }
        return list
    }

    public static func fetch() async throws -> Data {
        var request = URLRequest(url: url, timeoutInterval: 15)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw URLError(.badServerResponse)
        }
        _ = try decode(data)
        return data
    }

    /// The newest report from a Mac with `chip`, or else the newest report.
    public func compat(for titleId: String, chip: String?) -> Compat? {
        guard let title = titles[titleId] else { return nil }
        let report = title.reports.first { $0.chip == chip } ?? title.reports.first
        guard let report else { return Compat(status: title.status) }
        let environment = "\(report.chip), macOS \(report.macos), ps5slop \(report.ps5slop)"
        let note = [report.notes, "Reported on \(environment)."].compactMap { $0 }.joined(separator: " ")
        return Compat(status: report.status, note: note)
    }

    public func applied(to games: [GameEntry], chip: String?) -> [GameEntry] {
        games.map { game in
            var game = game
            game.compat = game.titleId.flatMap { compat(for: $0, chip: chip) }
            return game
        }
    }

    public static var localChip: String? {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }
}
