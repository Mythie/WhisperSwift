# Plan to Address Review Issues

Based on the review in `0002-review-gemini3.md`, this document outlines the proposed fixes for each issue.

---

## Required Issues

### 1. Fix `containsSpeech` Averaging Logic

**Problem**: The current `containsSpeech` function calculates RMS over the entire buffer. A 30-second buffer with 29s silence + 1s speech may average below the threshold, causing speech to be missed.

**Current Code** (`SilenceDetector.swift:164-172`):
```swift
public static func containsSpeech(in samples: [Float], threshold: Float = 0.01) -> Bool {
    guard !samples.isEmpty else { return false }
    let rms = calculateRMS(samples: samples, start: 0, count: samples.count)
    return rms >= threshold
}
```

**Proposed Fix**:
- [ ] Modify `containsSpeech` to scan the buffer in windows (e.g., 100ms) and return `true` if *any* window exceeds the threshold
- [ ] Add optional `windowDuration` parameter to control scanning granularity (default: 0.1s)
- [ ] Use the existing `calculateRMS` function for each window

**Implementation**:
```swift
public static func containsSpeech(
    in samples: [Float],
    threshold: Float = 0.01,
    windowDuration: TimeInterval = 0.1
) -> Bool {
    guard !samples.isEmpty else { return false }
    
    let windowSize = Int(windowDuration * sampleRate)
    guard windowSize > 0 else { return false }
    
    // Check each window - return true if ANY window has speech
    var i = 0
    while i < samples.count {
        let count = min(windowSize, samples.count - i)
        let rms = calculateRMS(samples: samples, start: i, count: count)
        if rms >= threshold {
            return true
        }
        i += windowSize
    }
    
    return false
}
```

**Files to modify**: `Sources/WhisperSwift/Audio/SilenceDetector.swift`

---

### 2. Increase or Configure Overlap Duration

**Problem**: The hardcoded 0.1s (100ms) overlap when forcing a split may be insufficient, causing the first word of the next chunk to be clipped.

**Current Code** (`StreamingTranscriber.swift:349`):
```swift
let overlapSamples = Int(0.1 * AudioProcessor.requiredSampleRate)
```

**Proposed Fix**:
- [ ] Add `overlapDuration` parameter to `StreamingTranscriber.init()` with a default of 0.5s
- [ ] Store it as a property and use it in `transcribeWithSilenceDetection`
- [ ] Document the parameter in the initializer

**Implementation**:

1. Add property to `StreamingTranscriber`:
```swift
/// Overlap duration in seconds when forcing audio splits.
private let overlapDuration: TimeInterval
```

2. Add initializer parameter:
```swift
public init(
    ...
    overlapDuration: TimeInterval = 0.5,  // New parameter
    ...
)
```

3. Update usage in `transcribeWithSilenceDetection`:
```swift
let overlapSamples = Int(overlapDuration * AudioProcessor.requiredSampleRate)
```

**Files to modify**: `Sources/WhisperSwift/Transcription/StreamingTranscriber.swift`

---

## Nice to Have Issues

### 3. Latency Optimization: "Most Recent" vs. "Longest" Silence

**Problem**: `findSilenceBreak` searches for the *longest* silence, but for real-time streaming, finding the *most recent valid* silence (closest to the end) would reduce latency.

**Current Behavior** (`SilenceDetector.swift:110-148`): Finds the longest silence gap in the search window.

**Proposed Fix**:
- [ ] Add a `FindSilenceStrategy` enum with cases `.longest` and `.mostRecent`
- [ ] Add `strategy` parameter to `findSilenceBreak` and `SilenceDetectorOptions`
- [ ] Default to `.mostRecent` for streaming use cases
- [ ] When using `.mostRecent`, return the first valid silence gap found (searching backwards)

**Implementation**:
```swift
public enum FindSilenceStrategy: Sendable {
    /// Find the longest silence gap (better accuracy, higher latency)
    case longest
    /// Find the most recent valid silence gap (lower latency)
    case mostRecent
}

// In SilenceDetectorOptions:
public var strategy: FindSilenceStrategy = .mostRecent
```

**Files to modify**: `Sources/WhisperSwift/Audio/SilenceDetector.swift`

---

### 4. Code Deduplication

**Problem**: The logic for converting raw segments to `TranscriptionSegment`, checking for duplicates, and yielding to the continuation is repeated in three places.

**Locations**:
- `transcribeWithVAD` (lines 292-298)
- `transcribeChunk` (lines 364-370)
- `processFinalAudio` (lines 395-401)

**Proposed Fix**:
- [ ] Extract into a private helper method `emitSegments(_ rawSegments: [RawSegment])`
- [ ] Replace all three occurrences with calls to this helper

**Implementation**:
```swift
/// Emits transcribed segments to the stream, handling deduplication.
private func emitSegments(_ rawSegments: [some Sequence<RawSegment>]) async {
    for rawSegment in rawSegments {
        let transcriptionSegment = TranscriptionSegment(from: rawSegment)
        if await stateManager.shouldEmit(text: transcriptionSegment.text) {
            segmentContinuation.yield(transcriptionSegment)
        }
    }
}
```

**Files to modify**: `Sources/WhisperSwift/Transcription/StreamingTranscriber.swift`

---

### 5. Sliding Window for RMS Calculation

**Problem**: The detector uses non-overlapping windows, which could miss silence events that straddle window boundaries.

**Current Behavior**: Windows step by `windowSize` with no overlap.

**Proposed Fix**:
- [ ] Add `windowOverlap` parameter to `SilenceDetectorOptions` (default: 0.5 for 50% overlap)
- [ ] Calculate step size as `windowSize * (1 - overlap)`
- [ ] Update `findSilenceBreak` and `findSpeechSegments` to use the step size

**Implementation**:
```swift
// In SilenceDetectorOptions:
public var windowOverlap: Float = 0.5  // 50% overlap

// In findSilenceBreak:
let stepSize = max(1, Int(Double(windowSize) * Double(1.0 - options.windowOverlap)))
// Change: i -= windowSize  ->  i -= stepSize
```

**Note**: The review mentions this is "less critical" due to the high 10ms window resolution. Consider if the added complexity is worth it.

**Files to modify**: `Sources/WhisperSwift/Audio/SilenceDetector.swift`

---

### 6. Thread Safety Verification

**Problem**: Verify that `segmentContinuation` is accessed safely under Swift 6 strict concurrency.

**Current Implementation**: 
- `StreamingTranscriber` is marked `Sendable`
- `segmentContinuation` is `AsyncThrowingStream.Continuation` which is `Sendable`
- All access happens through `async` methods

**Proposed Fix**:
- [ ] Enable Swift 6 language mode or strict concurrency checking in Package.swift
- [ ] Run build and fix any concurrency warnings
- [ ] Verify no data races with Thread Sanitizer

**Verification Steps**:
1. Add to Package.swift:
```swift
swiftLanguageVersions: [.v6]  // or use SwiftSettings
// OR
swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
```

2. Build and check for warnings
3. Run tests with TSAN enabled

**Files to modify**: `Package.swift` (for verification), potentially source files if issues found

---

## Implementation Priority

### Phase 1: Required (Data Integrity)
1. **Issue 1**: Fix `containsSpeech` - HIGH priority, risk of data loss
2. **Issue 2**: Configurable overlap - MEDIUM priority, affects accuracy

### Phase 2: Nice to Have (Quality)
3. **Issue 3**: Most recent silence strategy - improves latency
4. **Issue 4**: Code deduplication - maintainability
5. **Issue 5**: Sliding window - marginal accuracy improvement
6. **Issue 6**: Thread safety verification - validation

---

## Approval Checklist

Please approve each item to proceed:

- [x] **Issue 1**: Fix `containsSpeech` windowed scanning
- [ ] **Issue 2**: Make overlap duration configurable (default 0.5s)
- [x] **Issue 3**: Add "most recent" silence strategy option
- [ ] **Issue 4**: Extract segment emission helper method
- [ ] **Issue 5**: Add sliding window overlap option
- [x] **Issue 6**: Enable strict concurrency checking
