import Foundation

/// The last `capacity` lines of a child's output, fed with raw chunks that may split lines.
public struct LogBuffer: Sendable {
    public private(set) var lines: [String] = []
    public let capacity: Int
    private var partial = Data()

    public init(capacity: Int = 5000) {
        self.capacity = capacity
    }

    public mutating func append(_ chunk: Data) {
        partial.append(chunk)
        var newLines: [String] = []
        while let nl = partial.firstIndex(of: UInt8(ascii: "\n")) {
            newLines.append(String(decoding: partial[partial.startIndex..<nl], as: UTF8.self))
            partial.removeSubrange(partial.startIndex...nl)
        }
        if partial.count > 64 * 1024 {
            newLines.append(String(decoding: partial, as: UTF8.self))
            partial.removeAll()
        }
        add(newLines)
    }

    public mutating func appendLine(_ line: String) {
        add([line])
    }

    public mutating func flush() {
        guard !partial.isEmpty else { return }
        add([String(decoding: partial, as: UTF8.self)])
        partial.removeAll()
    }

    public mutating func clear() {
        lines.removeAll()
        partial.removeAll()
    }

    private mutating func add(_ new: [String]) {
        guard !new.isEmpty else { return }
        lines.append(contentsOf: new)
        if lines.count > capacity {
            lines.removeFirst(lines.count - capacity)
        }
    }
}
