# Beta Streaming Transcriber Implementation Plan

## Goal

Create a new `BetaStreamingTranscriber` that provides a cleaner API where the consumer passes the `AVAudioInputNode` directly and the transcriber handles all audio processing internally.

## Target API

```swift
let engine = AVAudioEngine()
let input = engine.inputNode

let transcriber = try await BetaStreamingTranscriber(
    modelPath: modelURL,
    input: input
    // ... other options
)

try await transcriber.start()

for try await segment in transcriber.segments {
    print(segment.text)
}

let finalResult = try await transcriber.stop()
print(finalResult.text)
```

## Key Differences from Current StreamingTranscriber

| Aspect | Current | Beta |
|--------|---------|------|
| Audio Input | Consumer calls `process(buffer:)` manually | Pass `AVAudioInputNode` at init, transcriber installs tap |
| Tap Management | Consumer responsibility | Transcriber manages tap lifecycle |
| Engine Control | Consumer manages entirely | Consumer manages engine, transcriber manages tap |

## Implementation Steps

### 1. Core Structure
- Create `BetaStreamingTranscriber` class conforming to `Sendable`
- Store `AVAudioInputNode` (which is `Sendable` as of iOS 17+/macOS 14+)
- Initialize `WhisperContext`, optionally `VADContext`
- Set up `AsyncThrowingStream` for segment emission

### 2. Audio Input Handling
- Install tap on the input node during `start()`
- Use appropriate buffer size (4096 samples is typical)
- Convert incoming buffers using `AudioProcessor.convert()`
- Accumulate in `AudioRingBuffer`

### 3. Transcription Loop
- Run background task that monitors audio buffer duration
- When `minAudioDuration` is reached:
  - If VAD available: detect speech segments, transcribe each
  - If no VAD: use `SilenceDetector` to find break points
- Emit segments through the continuation
- Handle deduplication via `StreamingStateManager`

### 4. Lifecycle Management
- `start()`: Install tap, begin processing loop
- `stop()`: Remove tap, process final audio, return complete result
- State machine: idle -> running -> stopping -> stopped

### 5. Thread Safety
- Use actors for all mutable state
- `AVAudioInputNode` tap callback is on audio thread - dispatch to async context
- Ensure `Sendable` conformance throughout

## Design Decisions

1. **Tap Installation**: The transcriber will install its own tap. If the consumer needs multiple taps, they should use a splitter pattern or the original `StreamingTranscriber`.

2. **Buffer Size**: Use 4096 samples (~256ms at 16kHz) as a reasonable default that balances latency and efficiency.

3. **Tap Format**: Request the hardware format and convert internally rather than forcing a specific format.

4. **Error Handling**: Errors during transcription will be emitted through the segment stream and also set the state to `.failed`.

5. **No Engine Management**: The transcriber does NOT start/stop the `AVAudioEngine`. The consumer is responsible for that. This keeps responsibilities clear.

## File Location

`Sources/WhisperSwift/Transcription/BetaStreamingTranscriber.swift`
