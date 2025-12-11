# Getting Started

Set up WhisperSwift and transcribe your first audio file.

## Overview

WhisperSwift requires a Whisper model file in GGML format. You can download pre-converted models from the whisper.cpp repository or convert your own.

## Download a Model

Download a model from the whisper.cpp releases. The "tiny" model is good for testing:

```bash
# Download the tiny English-only model (~75MB)
curl -L -o ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin

# Or the tiny multilingual model (~75MB)
curl -L -o ggml-tiny.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
```

Available model sizes:

| Model | English-only | Multilingual | Parameters |
|-------|-------------|--------------|------------|
| tiny | ggml-tiny.en.bin | ggml-tiny.bin | 39M |
| base | ggml-base.en.bin | ggml-base.bin | 74M |
| small | ggml-small.en.bin | ggml-small.bin | 244M |
| medium | ggml-medium.en.bin | ggml-medium.bin | 769M |
| large | - | ggml-large-v3.bin | 1550M |

> Note: English-only models are faster and more accurate for English audio.
> For other languages, use the multilingual models.

## Transcribe an Audio File

Create a ``Transcriber`` and transcribe a file:

```swift
import WhisperSwift

// Load the model
let modelURL = URL(fileURLWithPath: "/path/to/ggml-tiny.en.bin")
let transcriber = try await Transcriber(modelPath: modelURL)

// Transcribe an audio file
let audioURL = URL(fileURLWithPath: "/path/to/audio.wav")
let result = try await transcriber.transcribe(file: audioURL)

// Print the transcription
print(result.text)

// Access individual segments with timing
for segment in result.segments {
    print("[\(segment.startTime)s - \(segment.endTime)s]: \(segment.text)")
}
```

## Supported Audio Formats

WhisperSwift uses AVFoundation for audio loading and supports:

- WAV (recommended)
- MP3
- M4A / AAC
- CAF
- AIFF
- Any format supported by AVAudioFile

Audio is automatically converted to the required format (16kHz mono Float32).

## Configuration Options

Customize transcription behavior with ``TranscriptionOptions``:

```swift
let options = TranscriptionOptions(
    language: .english,        // Specify language (faster than auto-detect)
    translate: false,          // Set true to translate to English
    tokenTimestamps: true,     // Get word-level timing
    initialPrompt: "Meeting transcript:"  // Condition the model
)

let result = try await transcriber.transcribe(file: audioURL, options: options)
```

## Hardware Configuration

Control GPU and CPU usage with ``WhisperConfiguration``:

```swift
// Use GPU (default)
let transcriber = try await Transcriber(
    modelPath: modelURL,
    configuration: .default
)

// CPU only (useful for testing or debugging)
let cpuTranscriber = try await Transcriber(
    modelPath: modelURL,
    configuration: .cpuOnly
)

// Custom configuration
let config = WhisperConfiguration(
    useGPU: true,
    useFlashAttention: true,
    threadCount: 4
)
let customTranscriber = try await Transcriber(
    modelPath: modelURL,
    configuration: config
)
```

## Error Handling

WhisperSwift uses ``WhisperError`` for all error cases:

```swift
do {
    let transcriber = try await Transcriber(modelPath: modelURL)
    let result = try await transcriber.transcribe(file: audioURL)
} catch let error as WhisperError {
    switch error {
    case .modelNotFound(let url):
        print("Model not found at: \(url.path)")
    case .modelLoadFailed(let reason):
        print("Failed to load model: \(reason)")
    case .audioLoadFailed(let url, let underlying):
        print("Failed to load audio: \(url.path), \(underlying)")
    case .transcriptionFailed(let reason):
        print("Transcription failed: \(reason)")
    default:
        print("Error: \(error.localizedDescription)")
    }
}
```

## Next Steps

- Learn about real-time streaming in <doc:RealTimeStreaming>
- Explore all configuration options in ``TranscriptionOptions``
- See supported languages in ``Language``
