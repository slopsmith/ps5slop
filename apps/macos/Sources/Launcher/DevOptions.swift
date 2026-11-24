import AppKit
import ScreenCaptureKit

/// Launch options for screenshots: `--appearance light|dark`, `--window WxH`, `--about`,
/// and `--snapshot PATH`, which saves the window as a PNG once drawn and quits.
@MainActor
enum Snapshot {
    static func takeIfAsked() async {
        if let appearance = LauncherApp.argument("--appearance") {
            NSApp.appearance = NSAppearance(named: appearance == "dark" ? .darkAqua : .aqua)
        }
        var window = NSApp.windows.first { $0.isVisible }
        for _ in 0..<50 where window == nil {
            try? await Task.sleep(for: .milliseconds(100))
            window = NSApp.windows.first { $0.isVisible }
        }
        if let size = LauncherApp.argument("--window")?.split(separator: "x").compactMap({ Double($0) }),
            size.count == 2
        {
            window?.setContentSize(NSSize(width: size[0], height: size[1]))
            window?.center()
        }
        guard let path = LauncherApp.argument("--snapshot"), let window else { return }
        try? await Task.sleep(for: .seconds(2))
        let url = URL(fileURLWithPath: path)
        do {
            try await capture(window, to: url)
        } catch {
            FileHandle.standardError.write(
                Data("window capture failed (\(error.localizedDescription)); drawing without glass\n".utf8))
            drawWithoutWindowServer(window, to: url)
        }
        NSApp.terminate(nil)
    }

    /// Misses everything the window server composites: glass, sidebar and inspector backgrounds.
    private static func drawWithoutWindowServer(_ window: NSWindow, to url: URL) {
        guard let view = window.contentView?.superview,
            let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds)
        else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: url)
    }

    private static func capture(_ window: NSWindow, to url: URL) async throws {
        let content = try await SCShareableContent.currentProcess
        guard let target = content.windows.first(where: { $0.windowID == CGWindowID(window.windowNumber) }) else {
            throw CocoaError(.fileNoSuchFile)
        }
        let filter = SCContentFilter(desktopIndependentWindow: target)
        let config = SCStreamConfiguration()
        let scale = window.backingScaleFactor
        config.width = Int(filter.contentRect.width * scale)
        config.height = Int(filter.contentRect.height * scale)
        config.showsCursor = false
        let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        let rep = NSBitmapImageRep(cgImage: image)
        guard let png = rep.representation(using: .png, properties: [:]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        try png.write(to: url)
    }
}
