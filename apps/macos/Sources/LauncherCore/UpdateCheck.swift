import Foundation

public struct Nightly: Decodable, Equatable, Sendable {
    public var commit: String
    public var date: String
    public var download: URL
    public var release: URL

    public init(commit: String, date: String, download: URL, release: URL) {
        self.commit = commit
        self.date = date
        self.download = download
        self.release = release
    }

    public static func decode(_ data: Data) throws -> Nightly {
        try JSONDecoder().decode(Nightly.self, from: data)
    }
}

public enum UpdateCheck {
    public enum Status: Equatable, Sendable {
        case localBuild
        case upToDate
        case available(Nightly)
    }

    public static func isRelease(_ commit: String?) -> Bool {
        guard let commit, commit.count == 40 else { return false }
        return commit.allSatisfy { ("0"..."9").contains($0) || ("a"..."f").contains($0) }
    }

    public static func status(current: String?, nightly: Nightly) -> Status {
        guard isRelease(current) else { return .localBuild }
        return nightly.commit == current ? .upToDate : .available(nightly)
    }

    public static func feed(source: URL) -> URL {
        source.appending(path: "releases/download/nightly/nightly.json")
    }

    public static func fetch(source: URL) async throws -> Nightly {
        var request = URLRequest(url: feed(source: source), timeoutInterval: 15)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw URLError(.badServerResponse)
        }
        return try Nightly.decode(data)
    }
}
