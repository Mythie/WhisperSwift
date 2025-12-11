# Real-Time Streaming

Transcribe audio in real-time from a microphone or other audio source.

## Overview

The ``StreamingTranscriber`` provides real-time transcription by accumulating audio samples and emitting transcription segments as they become available. It works seamlessly with AVAudioEngine for microphone input.

## Basic Usage

Create a streaming transcriber and iterate over segments:

```swift
import WhisperSwift
import AVFoundation

// Create the transcriber
let modelURL = URL(fileURLWithPath: "/path/to/ggml-tiny.en.bin")
let transcriber = try await StreamingTranscriber(modelPath: modelURL)

// Start the streaming session
try await transcriber.start()

// Iterate over segments as they arrive
Task {
    for try await segment in transcriber.segments {
        print("[\(String(format: "%.1f", segment.startTime))s]: \(segment.text)")
    }
}

// When done, stop the transcriber
let finalResult = try await transcriber.stop()
```

## Integration with AVAudioEngine

Connect the transcriber to your audio engine's input:

```swift
import AVFoundation
import WhisperSwift

class AudioTranscriptionManager {
    private var audioEngine: AVAudioEngine!
    private var transcriber: StreamingTranscriber!
    
    func startTranscription(modelPath: URL) async throws {
        // Initialize audio engine
        audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        
        // Create and start transcriber
        transcriber = try await StreamingTranscriber(modelPath: modelPath)
        try await transcriber.start()
        
        // Install tap to capture audio
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            Task {
                try await self?.transcriber.process(buffer: buffer)
            }
        }
        
        // Start audio engine
        try audioEngine.start()
        
        // Process segments
        Task {
            for try await segment in transcriber.segments {
                await MainActor.run {
                    // Update UI with transcription
                    print(segment.text)
                }
            }
        }
    }
    
    func stopTranscription() async throws -> TranscriptionResult {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        return try await transcriber.stop()
    }
}
```

## Audio Chunking Strategies

WhisperSwift provides two strategies for breaking audio into transcribable chunks:

### Silence Detection (Default)

By default, ``StreamingTranscriber`` uses lightweight RMS-based silence detection to find natural break points in speech. This works well for most use cases and requires no additional models.

```swift
// Configure silence detection
let silenceOptions = SilenceDetectorOptions(
    threshold: 0.01,           // RMS threshold for silence
    minSilenceDuration: 0.3,   // Minimum silence gap (seconds)
    searchDuration: 5.0,       // How far back to search
    windowDuration: 0.01       // RMS window size (seconds)
)

let transcriber = try await StreamingTranscriber(
    modelPath: modelURL,
    silenceDetectorOptions: silenceOptions,
    minAudioDuration: 1.0,     // Minimum audio before transcription
    maxAudioDuration: 30.0     // Maximum audio before forced transcription
)
```

### Neural VAD (Optional)

For more accurate speech detection in challenging conditions (noise, music, etc.), you can provide a Silero VAD model:

```swift
let transcriber = try await StreamingTranscriber(
    modelPath: modelURL,
    vadModelPath: vadModelURL,  // Path to Silero VAD model
    vadOptions: VADOptions(
        threshold: 0.5,
        minSpeechDurationMs: 250,
        minSilenceDurationMs: 100
    )
)
```

Download the Silero VAD model:
```bash
curl -L -o silero-vad.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-silero-vad.bin
```

## State Management

Monitor the transcriber's state with ``StreamingState``:

```swift
let state = await transcriber.state

switch state {
case .idle:
    print("Ready to start")
case .running:
    print("Currently transcribing")
case .stopping:
    print("Finishing up...")
case .stopped:
    print("Transcription complete")
case .failed(let error):
    print("Error: \(error)")
}
```

## Processing Raw Samples

If you have raw audio samples (e.g., from a custom audio source), use ``StreamingTranscriber/process(samples:sampleRate:)``:

```swift
// Process samples from any source
let samples: [Float] = getAudioSamples()  // Your audio source
try await transcriber.process(samples: samples, sampleRate: 48000)
```

The samples will be automatically resampled to 16kHz if needed.

## Configuration Options

### Transcription Options

Control language and transcription behavior:

```swift
let options = TranscriptionOptions(
    language: .english,        // Faster than auto-detect
    translate: false,
    tokenTimestamps: false,
    initialPrompt: nil
)

let transcriber = try await StreamingTranscriber(
    modelPath: modelURL,
    options: options
)
```

### Hardware Configuration

Optimize for your hardware:

```swift
let config = WhisperConfiguration(
    useGPU: true,              // Metal acceleration
    useFlashAttention: true,   // Faster on Apple Silicon
    threadCount: 4             // CPU threads
)

let transcriber = try await StreamingTranscriber(
    modelPath: modelURL,
    configuration: config
)
```

## Best Practices

1. **Use English-only models for English audio** - They're faster and more accurate.

2. **Set the language explicitly** - Auto-detection adds latency.

3. **Use smaller models for real-time** - `tiny` or `base` models provide lower latency.

4. **Handle errors gracefully** - The segment stream can throw errors.

5. **Call stop() when done** - This processes remaining audio and cleans up resources.

```swift
defer {
    Task {
        _ = try? await transcriber.stop()
    }
}
```

## See Also

- ``StreamingTranscriber``
- ``StreamingState``
- ``SilenceDetectorOptions``
- ``VADOptions``
