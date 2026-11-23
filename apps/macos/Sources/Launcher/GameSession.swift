import Foundation
import LauncherCore
import Observation
import os

/// The one running game: a `ps5slop run` child process and its output.
@MainActor
@Observable
final class GameSession {
    private(set) var game: GameEntry?
    private(set) var log = LogBuffer(capacity: 5000)
    private(set) var lastError: String?
    var isRunning: Bool { process != nil }

    private var process: Process?
    private var controlPath: String?
    private let pending = OSAllocatedUnfairLock(initialState: Data())
    private var drainTask: Task<Void, Never>?

    func launch(_ game: GameEntry, dataFolder: String, fontsFolder: String) {
        guard process == nil else { return }
        guard let binary = BinaryLocator.ps5slop() else {
            lastError = "The ps5slop binary was not found. Set PS5SLOP_BIN or build target/release/ps5slop."
            return
        }
        let control = FileManager.default.temporaryDirectory
            .appendingPathComponent("ps5slop-launcher-\(ProcessInfo.processInfo.processIdentifier).sock").path
        var args = ["run"]
        if !dataFolder.isEmpty { args += ["--data", dataFolder] }
        if !fontsFolder.isEmpty { args += ["--fonts", fontsFolder] }
        args += ["--control", control, game.path]

        let child = Process()
        child.executableURL = binary
        child.arguments = args
        child.standardInput = FileHandle.nullDevice
        let pipe = Pipe()
        child.standardOutput = pipe
        child.standardError = pipe
        let pending = self.pending
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
            } else {
                pending.withLock { $0.append(data) }
            }
        }
        child.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            let reason = proc.terminationReason
            Task { @MainActor in self?.finished(status: status, reason: reason) }
        }

        log.clear()
        log.appendLine("$ \(binary.path) \(args.joined(separator: " "))")
        do {
            try child.run()
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            lastError = "Cannot start ps5slop: \(error.localizedDescription)"
            return
        }
        self.game = game
        process = child
        controlPath = control
        lastError = nil
        drainTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(150))
                self?.drain()
            }
        }
    }

    /// Asks the game to quit over its control socket, and sends SIGTERM if
    /// that fails or the game is still running after a few seconds.
    func stop() {
        guard let child = process else { return }
        let pid = child.processIdentifier
        let path = controlPath
        Task.detached {
            let asked = path.map { ControlSocket.send("quit", to: $0) } ?? false
            if asked {
                for _ in 0..<30 where child.isRunning {
                    try? await Task.sleep(for: .milliseconds(100))
                }
            }
            if child.isRunning, child.processIdentifier == pid {
                kill(pid, SIGTERM)
            }
        }
    }

    func clearLog() {
        log.clear()
    }

    private func drain() {
        let data = pending.withLock { buffer in
            defer { buffer.removeAll() }
            return buffer
        }
        if !data.isEmpty { log.append(data) }
    }

    private func finished(status: Int32, reason: Process.TerminationReason) {
        drainTask?.cancel()
        drainTask = nil
        drain()
        let how = reason == .uncaughtSignal ? "killed by signal \(status)" : "exited with status \(status)"
        Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard let self else { return }
            drain()
            log.flush()
            log.appendLine("ps5slop \(how)")
        }
        process = nil
        controlPath = nil
        game = nil
    }
}
