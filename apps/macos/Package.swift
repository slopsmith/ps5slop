// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ps5slop-launcher",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "ps5slop-launcher", targets: ["Launcher"])
    ],
    targets: [
        .target(name: "LauncherCore"),
        .executableTarget(name: "Launcher", dependencies: ["LauncherCore"]),
        .testTarget(
            name: "LauncherCoreTests",
            dependencies: ["LauncherCore"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
