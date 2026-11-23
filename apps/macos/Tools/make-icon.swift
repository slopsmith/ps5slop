// Draws the app icon into an .iconset folder: xcrun swift make-icon.swift OUT.iconset
import AppKit

let out = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? "AppIcon.iconset")
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

func draw(_ px: Int) -> Data? {
    let s = CGFloat(px)
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8,
        samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0)
    else { return nil }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

    let inset = s * 0.1
    let body = NSRect(x: inset, y: inset, width: s - 2 * inset, height: s - 2 * inset)
    let shape = NSBezierPath(roundedRect: body, xRadius: body.width * 0.225, yRadius: body.width * 0.225)
    NSGradient(
        starting: NSColor(calibratedRed: 0.98, green: 0.45, blue: 0.25, alpha: 1),
        ending: NSColor(calibratedRed: 0.62, green: 0.16, blue: 0.55, alpha: 1))?
        .draw(in: shape, angle: -60)

    let drip = NSBezierPath()
    let w = body.width
    drip.move(to: NSPoint(x: body.minX, y: body.minY + w * 0.38))
    drip.curve(
        to: NSPoint(x: body.maxX, y: body.minY + w * 0.30),
        controlPoint1: NSPoint(x: body.minX + w * 0.35, y: body.minY + w * 0.12),
        controlPoint2: NSPoint(x: body.minX + w * 0.6, y: body.minY + w * 0.5))
    drip.line(to: NSPoint(x: body.maxX, y: body.minY))
    drip.line(to: NSPoint(x: body.minX, y: body.minY))
    drip.close()
    NSGraphicsContext.current?.saveGraphicsState()
    shape.addClip()
    NSColor.white.withAlphaComponent(0.18).setFill()
    drip.fill()
    NSGraphicsContext.current?.restoreGraphicsState()

    let text = NSAttributedString(
        string: "5",
        attributes: [
            .font: NSFont.systemFont(ofSize: w * 0.62, weight: .black),
            .foregroundColor: NSColor.white,
        ])
    let size = text.size()
    text.draw(at: NSPoint(x: body.midX - size.width / 2, y: body.midY - size.height / 2 + w * 0.04))

    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])
}

for points in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = scale == 1 ? "icon_\(points)x\(points).png" : "icon_\(points)x\(points)@2x.png"
        guard let png = draw(points * scale) else { fatalError("cannot draw \(name)") }
        try png.write(to: out.appendingPathComponent(name))
    }
}
