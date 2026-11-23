// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ps5slop-launcher",
    platforms: [.macOS(.v26)],
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
