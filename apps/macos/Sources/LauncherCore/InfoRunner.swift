import Foundation
import os

public struct InfoError: Error, LocalizedError, Sendable {
    public var message: String
    public var errorDescription: String? { message }
}

/// Runs `ps5slop info --json --icons DIR PATH...` and decodes its output.
public struct InfoRunner: Sendable {
    public var binary: URL
    public var iconsFolder: URL

    public init(binary: URL, iconsFolder: URL) {
        self.binary = binary
        self.iconsFolder = iconsFolder
    }

    public func scan(_ paths: [URL]) async throws -> [GameEntry] {
        if paths.isEmpty { return [] }
        try FileManager.default.createDirectory(at: iconsFolder, withIntermediateDirectories: true)
        let args = ["info", "--json", "--icons", iconsFolder.path] + paths.map(\.path)
        let result = try await ChildOutput.run(binary, arguments: args)
        guard result.status == 0 else {
            let detail = String(decoding: result.stderr, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw InfoError(message: "ps5slop info failed (\(result.status)): \(detail)")
        }
        return try GameInfoDecoder.decode(result.stdout)
    }
}

struct ChildOutput: Sendable {
    var status: Int32
    var stdout: Data
    var stderr: Data

    static func run(_ binary: URL, arguments: [String]) async throws -> ChildOutput {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = binary
                process.arguments = arguments
                let out = Pipe()
                let err = Pipe()
                process.standardOutput = out
                process.standardError = err
                process.standardInput = FileHandle.nullDevice
                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                    return
                }
                let group = DispatchGroup()
                let errData = OSAllocatedUnfairLock(initialState: Data())
                group.enter()
                DispatchQueue.global().async {
                    let data = err.fileHandleForReading.readDataToEndOfFile()
                    errData.withLock { $0 = data }
                    group.leave()
                }
                let outData = out.fileHandleForReading.readDataToEndOfFile()
                group.wait()
                process.waitUntilExit()
                continuation.resume(
                    returning: ChildOutput(
                        status: process.terminationStatus, stdout: outData, stderr: errData.withLock { $0 }))
            }
        }
    }
}
