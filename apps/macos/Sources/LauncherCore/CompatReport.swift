import Foundation

// Mirrors `.github/scripts/check.js` in slopsmith/ps5slop-compat. Keep the two in step.
public struct CompatReport: Sendable, Equatable {
    public enum Question: Int, CaseIterable, Sendable {
        case onScreen, menus, gameplay, playedThrough, broken, playedFully

        public var title: String {
            switch self {
            case .onScreen: "Does the game show anything on screen?"
            case .menus: "Did you reach the title screen or a menu?"
            case .gameplay: "Did you get into the game itself?"
            case .playedThrough: "Could you keep playing without the game stopping you?"
            case .broken: "Did anything look, sound or play wrong?"
            case .playedFully: "Did you play the whole game?"
            }
        }

        public var detail: String {
            switch self {
            case .onScreen:
                "Answer yes if the game draws anything of its own, even for a moment: a logo, a warning, a loading screen, or a garbled picture. A window that stays black, or a game that quits before anything appears, is a no."
            case .menus:
                "Answer yes if you got to the title screen or the main menu and could move around in it. Logos and loading screens on their own are a no."
            case .gameplay:
                "Answer yes if you started actually playing: controlling a character, a car, a board, whatever the game is about. Intros, cutscenes and menus are not gameplay."
            case .playedThrough:
                "Answer yes if you played to the end, or, for a game without an end, for a long session of an hour or more, and no crash, freeze or bug blocked you. If something keeps the game from being played properly, answer no."
            case .broken:
                "Answer yes if you noticed any problem, however small: graphical glitches, missing effects, crackling or missing sound, stutter, slowdown, or controls that misbehave."
            case .playedFully:
                "Answer yes if you played from the start to the end, so you can say the whole game works. If you only played part of it, answer no."
            }
        }

        var field: String {
            switch self {
            case .onScreen: "on_screen"
            case .menus: "menus"
            case .gameplay: "gameplay"
            case .playedThrough: "played_through"
            case .broken: "broken"
            case .playedFully: "played_fully"
            }
        }
    }

    public static let form = URL(string: "https://github.com/slopsmith/ps5slop-compat/issues/new")!

    public var titleId: String
    public var gameName: String?
    public var gameVersion: String?
    public var answers: [Question: Bool] = [:]
    public var notes = ""
    public var macos: String
    public var chip: String
    public var ps5slop: String

    public init(titleId: String, gameName: String?, gameVersion: String?, macos: String, chip: String, ps5slop: String) {
        self.titleId = titleId
        self.gameName = gameName
        self.gameVersion = gameVersion
        self.macos = macos
        self.chip = chip
        self.ps5slop = ps5slop
    }

    public func isAsked(_ question: Question) -> Bool {
        Question.allCases.prefix(min(question.rawValue, Question.broken.rawValue)).allSatisfy { answers[$0] == true }
    }

    public var askedQuestions: [Question] { Question.allCases.filter(isAsked) }

    public func question(after question: Question) -> Question? {
        askedQuestions.first { $0.rawValue > question.rawValue }
    }

    public func question(before question: Question) -> Question? {
        askedQuestions.last { $0.rawValue < question.rawValue }
    }

    public var status: CompatStatus? {
        guard askedQuestions.allSatisfy({ answers[$0] != nil }) else { return nil }
        let yes = { (q: Question) in answers[q] == true }
        if !yes(.onScreen) { return .nothing }
        if !yes(.menus) { return .boots }
        if !yes(.gameplay) { return .menus }
        if !yes(.playedThrough) { return .ingame }
        if yes(.broken) || !yes(.playedFully) { return .playable }
        return .perfect
    }

    public var url: URL? {
        guard status != nil else { return nil }
        var fields: [(String, String)] = [
            ("template", "launcher-report.yml"),
            ("title_id", titleId),
        ]
        if let gameName { fields.append(("game_name", gameName)) }
        if let gameVersion { fields.append(("game_version", gameVersion)) }
        for question in Question.allCases {
            fields.append((question.field, isAsked(question) && answers[question] == true ? "yes" : "no"))
        }
        let notes = notes.trimmingCharacters(in: .whitespacesAndNewlines)
        if !notes.isEmpty { fields.append(("notes", notes)) }
        fields += [("macos", macos), ("chip", chip), ("ps5slop", ps5slop)]

        var components = URLComponents(url: Self.form, resolvingAgainstBaseURL: false)!
        components.percentEncodedQuery = fields.map { "\($0)=\(Self.encode($1))" }.joined(separator: "&")
        return components.url
    }

    private static func encode(_ value: String) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        return value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
    }

    public static var localMacOS: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return v.patchVersion == 0 ? "\(v.majorVersion).\(v.minorVersion)" : "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }
}
