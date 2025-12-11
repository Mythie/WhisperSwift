# WhisperSwift

Idiomatic Swift bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - high-quality speech-to-text transcription using OpenAI's Whisper model.

## Features

- **Batch transcription** - Transcribe audio files (WAV, MP3, M4A, CAF, etc.)
- **Real-time streaming** - Live transcription from AVAudioEngine / microphone
- **GPU acceleration** - Metal and CoreML support on Apple Silicon
- **99 languages** - Full Whisper language support with auto-detection
- **Lightweight chunking** - RMS-based silence detection (no VAD model required)
- **Optional neural VAD** - Silero VAD support for challenging audio
- **Swift 6 ready** - Full concurrency safety with actors and Sendable types

## Requirements

- macOS 13.3+
- Swift 6.0+
- Xcode 15+

## Installation

### Swift Package Manager

Add WhisperSwift to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Mythie/WhisperSwift.git", from: "1.0.0")
]
```

Or in Xcode: File → Add Package Dependencies → Enter the repository URL.

## Quick Start

### Download a Model

```bash
# Tiny English-only model (~75MB) - fast, good for testing
curl -L -o ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin

# Base English model (~140MB) - better accuracy
curl -L -o ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

### Batch Transcription

```swift
import WhisperSwift

// Load the model
let modelURL = Bundle.main.url(forResource: "ggml-tiny.en", withExtension: "bin")!
let transcriber = try await Transcriber(modelPath: modelURL)

// Transcribe an audio file
let audioURL = Bundle.main.url(forResource: "recording", withExtension: "wav")!
let result = try await transcriber.transcribe(file: audioURL)

print(result.text)
// "And so my fellow Americans, ask not what your country can do for you..."

// Access segments with timing
for segment in result.segments {
    print("[\(segment.startTime)s]: \(segment.text)")
}
```

### Real-Time Streaming

```swift
import WhisperSwift
import AVFoundation

// Create streaming transcriber
let transcriber = try await StreamingTranscriber(modelPath: modelURL)
try await transcriber.start()

// Set up AVAudioEngine
let audioEngine = AVAudioEngine()
let inputNode = audioEngine.inputNode
let format = inputNode.outputFormat(forBus: 0)

// Feed audio to transcriber
inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
    Task { try await transcriber.process(buffer: buffer) }
}

try audioEngine.start()

// Receive transcriptions as they arrive
for try await segment in transcriber.segments {
    print(segment.text)
}

// Clean up
audioEngine.stop()
let finalResult = try await transcriber.stop()
```

## Configuration

### Transcription Options

```swift
let options = TranscriptionOptions(
    language: .english,           // Specify language (faster than auto-detect)
    translate: false,             // Translate to English
    tokenTimestamps: true,        // Word-level timing
    initialPrompt: "Meeting notes:",  // Condition the model
    samplingStrategy: .greedy     // or .beamSearch(beamSize: 5)
)

let result = try await transcriber.transcribe(file: audioURL, options: options)
```

### Hardware Configuration

```swift
// Default: GPU + Flash Attention
let transcriber = try await Transcriber(
    modelPath: modelURL,
    configuration: .default
)

// CPU only
let cpuTranscriber = try await Transcriber(
    modelPath: modelURL,
    configuration: .cpuOnly
)

// Custom
let config = WhisperConfiguration(
    useGPU: true,
    useFlashAttention: true,
    threadCount: 4
)
```

### Streaming Options

```swift
// Configure silence detection (default chunking strategy)
let silenceOptions = SilenceDetectorOptions(
    threshold: 0.01,           // RMS threshold
    minSilenceDuration: 0.3,   // Minimum silence gap
    searchDuration: 5.0        // Search window
)

let transcriber = try await StreamingTranscriber(
    modelPath: modelURL,
    silenceDetectorOptions: silenceOptions,
    minAudioDuration: 1.0,     // Minimum before transcription
    maxAudioDuration: 30.0     // Maximum before forced transcription
)
```

## Supported Languages

WhisperSwift supports all 99 languages in Whisper:

```swift
// Specify a language
let options = TranscriptionOptions(language: .spanish)

// Auto-detect
let options = TranscriptionOptions(language: nil)  // or .auto

// Check detected language
if let detected = result.detectedLanguage {
    print("Detected: \(detected.displayName)")  // "Spanish"
}
```

See `Language` enum for the full list.

## Model Sizes

| Model | Parameters | English-only | Multilingual | Speed |
|-------|------------|--------------|--------------|-------|
| tiny | 39M | ggml-tiny.en.bin | ggml-tiny.bin | Fastest |
| base | 74M | ggml-base.en.bin | ggml-base.bin | Fast |
| small | 244M | ggml-small.en.bin | ggml-small.bin | Medium |
| medium | 769M | ggml-medium.en.bin | ggml-medium.bin | Slow |
| large | 1550M | - | ggml-large-v3.bin | Slowest |

> **Tip:** English-only models are faster and more accurate for English audio.

## Error Handling

```swift
do {
    let result = try await transcriber.transcribe(file: audioURL)
} catch let error as WhisperError {
    switch error {
    case .modelNotFound(let url):
        print("Model not found: \(url.path)")
    case .audioLoadFailed(let url, let underlying):
        print("Audio error: \(underlying)")
    case .transcriptionFailed(let reason):
        print("Transcription failed: \(reason)")
    default:
        print(error.localizedDescription)
    }
}
```

## Building the XCFramework

The pre-built XCFramework is included in `Frameworks/`. To rebuild from source:

```bash
# Clone whisper.cpp if not present
git clone https://github.com/ggerganov/whisper.cpp.git

# Build the framework
./scripts/build-xcframework-macos.sh

# Copy to Frameworks directory
cp -R whisper.cpp/build-apple/whisper.xcframework Frameworks/
```

This creates a universal (arm64 + x86_64) framework with Metal, CoreML, and BLAS support.

## License

MIT License - see LICENSE file.

## Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) by Georgi Gerganov
- [OpenAI Whisper](https://github.com/openai/whisper) by OpenAI
