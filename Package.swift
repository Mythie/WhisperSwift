// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "WhisperSwift",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "WhisperSwift",
            targets: ["WhisperSwift"]
        ),
    ],
    targets: [
        .target(
            name: "WhisperSwift",
            dependencies: ["whisper"],
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .binaryTarget(
            name: "whisper",
            path: "Frameworks/whisper.xcframework"
        ),
        .testTarget(
            name: "WhisperSwiftTests",
            dependencies: ["WhisperSwift"],
            resources: [
                .copy("Fixtures")
            ]
        ),
    ]
)
