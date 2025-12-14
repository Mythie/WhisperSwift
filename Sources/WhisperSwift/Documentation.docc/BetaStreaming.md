# Beta Streaming Transcription

Low-latency real-time transcription using a sliding window approach, modeled after whisper.cpp's stream example.

## Overview

``BetaStreamingTranscriber`` provides an alternative streaming approach that closely mirrors the [whisper.cpp stream example](https://github.com/ggerganov/whisper.cpp/tree/master/examples/stream). It uses a sliding window with overlap to deliver continuous transcription output with minimal latency.

### When to Use BetaStreamingTranscriber

| Use Case | Recommended Transcriber |
|----------|------------------------|
| Dictation with natural pauses | ``StreamingTranscriber`` |
| Continuous speech (lectures, meetings) | ``BetaStreamingTranscriber`` |
| Live captioning | ``BetaStreamingTranscriber`` |
| Voice commands with silence gaps | ``StreamingTranscriber`` |

**Key differences from ``StreamingTranscriber``:**

- **No silence detection** - Uses fixed time windows instead of waiting for speech pauses
- **Lower latency** - Emits partial transcriptions every few seconds
- **Simpler audio handling** - You provide an `AVAudioEngine`, it manages the rest
- **Matches whisper.cpp** - Same algorithm as the official stream example

## Basic Usage

```swift
import WhisperSwift
import AVFoundation

// Create and configure the audio engine
let engine = AVAudioEngine()

// Create the transcriber with your model
let transcriber = try await BetaStreamingTranscriber(
    modelPath: modelURL,
    engine: engine,
    configuration: .default
)

// Start the audio engine
try engine.start()

// Start transcription
try await transcriber.start()

// Consume segments as they arrive
for try await segment in transcriber.segments {
    print(segment.text)
    
    // segment.isPartial indicates if more text may come
    // segment.iteration is the processing window number
}

// When done, stop and get final text
let finalText = try await transcriber.stop()
```

## Operating Modes

BetaStreamingTranscriber supports two distinct operating modes:

### Sliding Window Mode (Default)

In sliding window mode, the transcriber processes audio in overlapping windows at fixed intervals. This provides consistent, low-latency output regardless of speech patterns.

```
Audio timeline:
|-------- window 1 --------|
              |-------- window 2 --------|
                            |-------- window 3 --------|
<--- step ---><--- step --->
```

**How it works:**

1. Audio accumulates in a buffer
2. Every `stepDuration` seconds, a `windowDuration` chunk is transcribed
3. A small `keepDuration` overlap prevents word boundary issues
4. Segments are emitted with `isPartial: true`

```swift
let config = BetaStreamingTranscriber.Configuration(
    stepDuration: 3.0,      // Process every 3 seconds
    windowDuration: 10.0,   // Use 10 seconds of context
    keepDuration: 0.2       // 200ms overlap
)
```

### VAD Mode

In VAD (Voice Activity Detection) mode, the transcriber uses the Silero neural VAD model to detect speech before transcribing. This produces higher quality output at the cost of latency.

To enable VAD mode, provide a path to the Silero VAD model:

```swift
// Download the Silero VAD model first:
// curl -L -o silero-vad.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-silero-vad.bin

let transcriber = try await BetaStreamingTranscriber(
    modelPath: modelURL,
    engine: engine,
    vadModelPath: vadModelURL,  // Enables VAD mode
    configuration: .vadMode
)
```

**How it works:**

1. Neural Silero VAD monitors for speech segments
2. When speech is detected, the audio segment is transcribed
3. Segments are emitted with `isPartial: false`

> Note: VAD currently runs on CPU only due to Metal compatibility issues.

## Configuration Presets

### Default Configuration

Matches the whisper.cpp stream example defaults. Good balance of latency and accuracy.

```swift
.default
// stepDuration: 3.0s
// windowDuration: 10.0s
// keepDuration: 0.2s
// maxTokens: 32
```

### Low Latency Configuration

Faster response times for interactive applications.

```swift
.lowLatency
// stepDuration: 1.0s
// windowDuration: 5.0s
// keepDuration: 0.2s
// maxTokens: 16
```

### VAD Mode Configuration

Wait for speech detection before transcribing. Requires a `vadModelPath` to be provided.

```swift
.vadMode
// keepDuration: 0.0 (not used in VAD mode)
// maxTokens: 0 (unlimited)
// vadOptions: .default
```

## Configuration Options

Create a custom configuration for fine-grained control:

```swift
let config = BetaStreamingTranscriber.Configuration(
    // Timing
    stepDuration: 2.0,           // New audio per iteration
    windowDuration: 8.0,         // Total context window
    keepDuration: 0.3,           // Overlap for continuity
    bufferDuration: 0.1,         // Audio callback interval
    
    // Whisper settings
    transcriptionOptions: TranscriptionOptions(
        language: .english,
        tokenTimestamps: false
    ),
    whisperConfiguration: WhisperConfiguration(
        useGPU: true,
        threadCount: 4
    ),
    
    // Context continuity
    keepContext: false,          // Carry tokens between iterations
    maxTokens: 32,               // Max tokens per chunk
    
    // VAD settings (only used when vadModelPath is provided)
    vadOptions: VADOptions(
        threshold: 0.5,
        minSpeechDurationMs: 250,
        minSilenceDurationMs: 100
    )
)
```

### Timing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stepDuration` | 3.0s | How often to process new audio |
| `windowDuration` | 10.0s | Total audio context for each transcription |
| `keepDuration` | 0.2s | Overlap to avoid cutting words |
| `bufferDuration` | 0.1s | Audio callback frequency |

### Context Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keepContext` | false | Use previous output as prompt for next iteration |
| `maxTokens` | 32 | Maximum tokens per transcription (0 = unlimited) |

### VAD Options

VAD mode is enabled by providing a `vadModelPath` to the initializer. Configure the VAD behavior with ``VADOptions``:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Speech probability threshold (0-1) |
| `minSpeechDurationMs` | 250 | Minimum speech duration to trigger |
| `minSilenceDurationMs` | 100 | Minimum silence to end speech segment |

## Complete Example

Here's a full implementation for a live transcription feature:

```swift
import SwiftUI
import AVFoundation
import WhisperSwift

@MainActor
class TranscriptionManager: ObservableObject {
    @Published var transcribedText = ""
    @Published var isRecording = false
    
    private var engine: AVAudioEngine?
    private var transcriber: BetaStreamingTranscriber?
    private var transcriptionTask: Task<Void, Never>?
    
    func startRecording(modelPath: URL) async throws {
        // Set up audio session (iOS)
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)
        #endif
        
        // Create audio engine
        let engine = AVAudioEngine()
        self.engine = engine
        
        // Create transcriber with low-latency config
        let transcriber = try await BetaStreamingTranscriber(
            modelPath: modelPath,
            engine: engine,
            configuration: .lowLatency
        )
        self.transcriber = transcriber
        
        // Start audio engine
        try engine.start()
        
        // Start transcription
        try await transcriber.start()
        isRecording = true
        
        // Consume segments in background
        transcriptionTask = Task {
            do {
                for try await segment in transcriber.segments {
                    self.transcribedText = segment.text
                }
            } catch {
                print("Transcription error: \(error)")
            }
        }
    }
    
    func stopRecording() async throws -> String {
        transcriptionTask?.cancel()
        
        let finalText = try await transcriber?.stop() ?? ""
        
        engine?.stop()
        engine = nil
        transcriber = nil
        
        isRecording = false
        return finalText
    }
}
```

## Streaming Segments

Each segment emitted by the transcriber contains:

```swift
public struct StreamingSegment: Sendable {
    /// The transcribed text
    public let text: String
    
    /// Whether this is partial (may be updated) or final
    public let isPartial: Bool
    
    /// The processing iteration number
    public let iteration: Int
}
```

In sliding window mode, segments are always partial (`isPartial: true`) because the text may be refined in subsequent windows. In VAD mode, segments are final (`isPartial: false`).

## Best Practices

### Model Selection

Use smaller models for real-time transcription:

- **tiny** or **tiny.en** - Fastest, lowest accuracy
- **base** or **base.en** - Good balance for real-time
- **small** or **small.en** - Better accuracy, may lag on older devices

English-only models (`.en`) are faster and more accurate for English audio.

### Language Configuration

Always set the language explicitly to avoid auto-detection latency:

```swift
let options = TranscriptionOptions(language: .english)
```

### Handling Partial Results

In sliding window mode, each segment may overlap with previous ones. For display, simply replace the text:

```swift
for try await segment in transcriber.segments {
    // Just show the latest - it includes context from previous audio
    displayText = segment.text
}
```

### Resource Cleanup

Always stop the transcriber and engine when done:

```swift
defer {
    Task {
        _ = try? await transcriber.stop()
        engine.stop()
    }
}
```

### GPU Acceleration

Enable Metal for faster processing on Apple Silicon:

```swift
let config = WhisperConfiguration(
    useGPU: true,
    useFlashAttention: true
)
```

## Troubleshooting

### High Latency

- Reduce `stepDuration` (e.g., 1.0s)
- Reduce `windowDuration` (e.g., 5.0s)
- Use a smaller model (tiny or base)
- Enable GPU acceleration

### Missing Words at Boundaries

- Increase `keepDuration` (e.g., 0.5s)
- Increase `windowDuration` for more context

### Repeated or Hallucinated Text

- Set `keepContext: false` (default)
- Reduce `maxTokens`
- Use a larger model for better accuracy

### Audio Not Being Captured

- Ensure the `AVAudioEngine` is started before calling `start()`
- Check microphone permissions
- Verify the input node format is valid

## See Also

- ``BetaStreamingTranscriber``
- ``StreamingTranscriber``
- ``TranscriptionOptions``
- ``WhisperConfiguration``
- ``VADOptions``
