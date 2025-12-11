# ``WhisperSwift``

Idiomatic Swift bindings for whisper.cpp - high-quality speech-to-text transcription using OpenAI's Whisper model.

## Overview

WhisperSwift provides a modern Swift API for transcribing audio using the Whisper speech recognition model. It supports both batch file transcription and real-time streaming from microphone input.

### Features

- **Batch transcription** - Transcribe audio files (WAV, MP3, M4A, CAF, etc.)
- **Real-time streaming** - Live transcription from AVAudioEngine
- **GPU acceleration** - Metal and CoreML support on Apple Silicon
- **99 languages** - Full Whisper language support with auto-detection
- **Swift 6 ready** - Full concurrency safety with actors and Sendable types

### Quick Start

Transcribe an audio file:

```swift
import WhisperSwift

let transcriber = try await Transcriber(modelPath: modelURL)
let result = try await transcriber.transcribe(file: audioURL)
print(result.text)
```

Real-time streaming from microphone:

```swift
let transcriber = try await StreamingTranscriber(modelPath: modelURL)
try await transcriber.start()

// Feed audio from AVAudioEngine
audioEngine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
    Task { try await transcriber.process(buffer: buffer) }
}

// Receive transcription segments
for try await segment in transcriber.segments {
    print("[\(segment.startTime)s]: \(segment.text)")
}
```

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:RealTimeStreaming>

### Transcription

- ``Transcriber``
- ``StreamingTranscriber``
- ``TranscriptionResult``
- ``TranscriptionSegment``

### Configuration

- ``WhisperConfiguration``
- ``TranscriptionOptions``
- ``SamplingStrategy``
- ``Language``

### Streaming

- ``StreamingState``
- ``SilenceDetectorOptions``
- ``VADOptions``

### Audio Processing

- ``AudioProcessor``
- ``SilenceDetector``
- ``SpeechSegment``

### Results

- ``Token``
- ``TranscriptionTimings``

### Errors

- ``WhisperError``
