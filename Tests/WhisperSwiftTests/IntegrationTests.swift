import Testing
import Foundation
@testable import WhisperSwift

// MARK: - Test Fixtures

/// Shared test fixture paths and utilities
enum TestFixtures {
    /// Path to the fixtures directory
    static var fixturesURL: URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures")
    }
    
    /// Path to the test model (tiny.en - English only)
    static var modelPath: URL {
        fixturesURL.appendingPathComponent("ggml-tiny.en.bin")
    }
    
    /// Check if the test model exists
    static var hasModel: Bool {
        FileManager.default.fileExists(atPath: modelPath.path)
    }
    
    /// Path to the large multilingual model (for language detection tests)
    static var largeModelPath: URL {
        fixturesURL.appendingPathComponent("ggml-large-v3-turbo-q8_0.bin")
    }
    
    /// Check if the large model exists
    static var hasLargeModel: Bool {
        FileManager.default.fileExists(atPath: largeModelPath.path)
    }
    
    /// Path to Silero VAD v5 model
    static var vadV5ModelPath: URL {
        fixturesURL.appendingPathComponent("ggml-silero-v5.1.2.bin")
    }
    
    /// Check if VAD v5 model exists
    static var hasVADv5Model: Bool {
        FileManager.default.fileExists(atPath: vadV5ModelPath.path)
    }
    
    /// Path to Silero VAD v6 model
    static var vadV6ModelPath: URL {
        fixturesURL.appendingPathComponent("ggml-silero-v6.2.0.bin")
    }
    
    /// Check if VAD v6 model exists
    static var hasVADv6Model: Bool {
        FileManager.default.fileExists(atPath: vadV6ModelPath.path)
    }
    
    /// Path to JFK sample audio (in fixtures directory)
    static var jfkAudioPath: URL {
        fixturesURL.appendingPathComponent("jfk.wav")
    }
    
    /// Check if JFK audio exists
    static var hasJFKAudio: Bool {
        FileManager.default.fileExists(atPath: jfkAudioPath.path)
    }
}

/// Integration tests that require a real whisper model and audio files.
///
/// To run these tests, you need to:
/// 1. Download a whisper model (e.g., ggml-base.en.bin or ggml-tiny.en.bin)
/// 2. Place it in Tests/WhisperSwiftTests/Fixtures/
///
/// Download models from:
/// https://huggingface.co/ggerganov/whisper.cpp/tree/main
///
/// Quick download (tiny.en model, ~75MB):
/// ```
/// curl -L -o Tests/WhisperSwiftTests/Fixtures/ggml-tiny.en.bin \
///   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
/// ```
///
/// The JFK sample audio is already included in whisper.cpp/samples/jfk.wav
@Suite("Integration Tests", .enabled(if: TestFixtures.hasModel))
struct IntegrationTests {
    
    // MARK: - Model Loading Tests
    
    @Test("Can load whisper model")
    func loadModel() async throws {
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        _ = transcriber  // Just verify it loaded
    }
    
    @Test("Can load model with CPU-only configuration")
    func loadModelCPUOnly() async throws {
        let transcriber = try await Transcriber(
            modelPath: TestFixtures.modelPath,
            configuration: .cpuOnly
        )
        _ = transcriber
    }
    
    // MARK: - Transcription Tests
    
    @Test("Can transcribe JFK audio file")
    func transcribeJFKAudio() async throws {
        // Skip if JFK audio doesn't exist
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found at \(TestFixtures.jfkAudioPath.path)")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        // Use English explicitly since we're using an English-only model (tiny.en)
        let options = TranscriptionOptions(language: .english)
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        
        // The JFK audio says: "And so my fellow Americans, ask not what your 
        // country can do for you, ask what you can do for your country."
        #expect(!result.text.isEmpty)
        #expect(result.text.lowercased().contains("ask"))
        #expect(result.text.lowercased().contains("country"))
        #expect(result.segments.count > 0)
        
        // Check segment timing
        for segment in result.segments {
            #expect(segment.startTime >= 0)
            #expect(segment.endTime > segment.startTime)
        }
    }
    
    @Test("Transcription includes timing information")
    func transcriptionHasTimings() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        let options = TranscriptionOptions(language: .english)
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        
        // Timings should be available
        #expect(result.timings != nil)
        if let timings = result.timings {
            #expect(timings.totalMs > 0)
            #expect(timings.encodeMs > 0)
            #expect(timings.decodeMs > 0)
        }
    }
    
    @Test("Can transcribe with token timestamps")
    func transcribeWithTokenTimestamps() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        let options = TranscriptionOptions(language: .english, tokenTimestamps: true)
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        
        // Should have tokens with timestamps
        #expect(!result.segments.isEmpty)
        if let firstSegment = result.segments.first {
            #expect(firstSegment.tokens != nil)
            if let tokens = firstSegment.tokens {
                #expect(!tokens.isEmpty)
            }
        }
    }
    
    @Test("Can transcribe with beam search")
    func transcribeWithBeamSearch() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        let options = TranscriptionOptions(language: .english, samplingStrategy: .beamSearch(beamSize: 3))
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        
        #expect(!result.text.isEmpty)
        #expect(result.text.lowercased().contains("country"))
    }
    
    @Test("Can transcribe with initial prompt")
    func transcribeWithInitialPrompt() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        let options = TranscriptionOptions(language: .english, initialPrompt: "President Kennedy speaking:")
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        
        #expect(!result.text.isEmpty)
    }
    
    @Test("Empty audio returns empty result")
    func transcribeEmptyAudio() async throws {
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        let result = try await transcriber.transcribe(samples: [], options: .default)
        
        #expect(result.text.isEmpty)
        #expect(result.segments.isEmpty)
    }
    
    @Test("Can transcribe raw samples")
    func transcribeRawSamples() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Load audio using AudioProcessor
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        #expect(!samples.isEmpty)
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        let options = TranscriptionOptions(language: .english)
        let result = try await transcriber.transcribe(samples: samples, options: options)
        
        #expect(!result.text.isEmpty)
        #expect(result.text.lowercased().contains("country"))
    }
    
    // MARK: - Audio Processing Tests
    
    @Test("AudioProcessor loads WAV file correctly")
    func loadWAVFile() throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        // JFK audio is about 11 seconds at 16kHz = ~176,000 samples
        #expect(samples.count > 100000)
        #expect(samples.count < 200000)
        
        // Samples should be normalized to -1.0...1.0
        let maxAbs = samples.map { abs($0) }.max() ?? 0
        #expect(maxAbs <= 1.0)
    }
    
    @Test("AudioProcessor loads MP3 file correctly")
    func loadMP3File() throws {
        let mp3Path = TestFixtures.jfkAudioPath
            .deletingLastPathComponent()
            .appendingPathComponent("jfk.mp3")
        
        guard FileManager.default.fileExists(atPath: mp3Path.path) else {
            Issue.record("JFK MP3 not found")
            return
        }
        
        let samples = try AudioProcessor.loadAudioFile(mp3Path)
        
        // Should have similar sample count to WAV
        #expect(samples.count > 100000)
        
        // Samples should be normalized
        let maxAbs = samples.map { abs($0) }.max() ?? 0
        #expect(maxAbs <= 1.0)
    }
}

// MARK: - Large Model Tests (Multilingual)

@Suite("Large Model Tests", .enabled(if: TestFixtures.hasLargeModel))
struct LargeModelTests {
    
    @Test("Can load large model")
    func loadLargeModel() async throws {
        let transcriber = try await Transcriber(modelPath: TestFixtures.largeModelPath)
        _ = transcriber
    }
    
    @Test("Auto language detection works with nil language")
    func autoDetectLanguageNil() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.largeModelPath)
        // Use nil for language to trigger auto-detection
        let options = TranscriptionOptions(language: nil)
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        
        print("Auto-detect (nil) result:")
        print("  Text: \(result.text)")
        print("  Detected language: \(result.detectedLanguage?.displayName ?? "none")")
        
        #expect(!result.text.isEmpty, "Transcription should not be empty")
        #expect(result.text.lowercased().contains("country") || result.text.lowercased().contains("ask"), 
                "Transcription should contain expected words")
        
        // Should detect English
        #expect(result.detectedLanguage == .english, 
                "Should detect English, got: \(result.detectedLanguage?.displayName ?? "nil")")
    }
    
    @Test("Auto language detection works with .auto language")
    func autoDetectLanguageAuto() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.largeModelPath)
        // Use .auto explicitly for auto-detection
        let options = TranscriptionOptions(language: .auto)
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        
        print("Auto-detect (.auto) result:")
        print("  Text: \(result.text)")
        print("  Detected language: \(result.detectedLanguage?.displayName ?? "none")")
        
        #expect(!result.text.isEmpty, "Transcription should not be empty")
        #expect(result.text.lowercased().contains("country") || result.text.lowercased().contains("ask"), 
                "Transcription should contain expected words")
        
        // Should detect English
        #expect(result.detectedLanguage == .english, 
                "Should detect English, got: \(result.detectedLanguage?.displayName ?? "nil")")
    }
    
    @Test("Default options use auto-detection")
    func defaultOptionsAutoDetect() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.largeModelPath)
        // Use default options - should auto-detect
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: .default)
        
        print("Default options result:")
        print("  Text: \(result.text)")
        print("  Detected language: \(result.detectedLanguage?.displayName ?? "none")")
        
        #expect(!result.text.isEmpty, "Transcription should not be empty")
        
        // Should detect English
        #expect(result.detectedLanguage == .english, 
                "Should detect English, got: \(result.detectedLanguage?.displayName ?? "nil")")
    }
    
    @Test("Explicit language skips detection")
    func explicitLanguageSkipsDetection() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.largeModelPath)
        // Use explicit English
        let options = TranscriptionOptions(language: .english)
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        
        print("Explicit English result:")
        print("  Text: \(result.text)")
        print("  Detected language: \(result.detectedLanguage?.displayName ?? "none")")
        
        #expect(!result.text.isEmpty, "Transcription should not be empty")
        
        // When explicit language is set, detectedLanguage should be nil
        #expect(result.detectedLanguage == nil, 
                "Detected language should be nil when explicit language is set")
    }
}

// MARK: - Performance Tests

@Suite("Performance Tests", .enabled(if: TestFixtures.hasModel))
struct PerformanceTests {
    
    @Test("Transcription completes in reasonable time")
    func transcriptionPerformance() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        let options = TranscriptionOptions(language: .english)
        
        let start = Date()
        let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
        let elapsed = Date().timeIntervalSince(start)
        
        // JFK audio is ~11 seconds
        // With GPU, transcription should be faster than real-time
        // With CPU tiny model, should still be under 30 seconds
        #expect(elapsed < 30.0, "Transcription took \(elapsed)s, expected < 30s")
        #expect(!result.text.isEmpty)
        
        print("Transcription completed in \(String(format: "%.2f", elapsed))s")
        if let timings = result.timings {
            print("  Encode: \(String(format: "%.1f", timings.encodeMs))ms")
            print("  Decode: \(String(format: "%.1f", timings.decodeMs))ms")
            print("  Total:  \(String(format: "%.1f", timings.totalMs))ms")
        }
    }
    
    @Test("Multiple transcriptions can reuse model")
    func multipleTranscriptions() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        let transcriber = try await Transcriber(modelPath: TestFixtures.modelPath)
        let options = TranscriptionOptions(language: .english)
        
        // Run transcription 3 times with the same model
        for i in 1...3 {
            let result = try await transcriber.transcribe(file: TestFixtures.jfkAudioPath, options: options)
            #expect(!result.text.isEmpty, "Transcription \(i) failed")
        }
    }
}

// MARK: - VAD Tests (Silero v5)

@Suite("VAD v5 Tests", .enabled(if: TestFixtures.hasVADv5Model && TestFixtures.hasModel))
struct VADv5Tests {
    
    @Test("Can load Silero VAD v5 model")
    func loadVADv5Model() async throws {
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV5ModelPath, useGPU: false)
        _ = vad
    }
    
    @Test("VAD v5 detects speech in JFK audio")
    func vadV5DetectsSpeech() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV5ModelPath, useGPU: false)
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        let hasSpeech = await vad.detectSpeech(samples: samples)
        #expect(hasSpeech == true, "VAD v5 should detect speech in JFK audio")
    }
    
    @Test("VAD v5 returns speech segments")
    func vadV5GetsSpeechSegments() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV5ModelPath, useGPU: false)
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        let segments = try await vad.getSpeechSegments(samples: samples, options: .default)
        
        #expect(!segments.isEmpty, "VAD v5 should return speech segments")
        
        // Check segment timing makes sense
        for segment in segments {
            #expect(segment.startTime >= 0, "Start time should be non-negative")
            #expect(segment.endTime > segment.startTime, "End time should be after start time")
            #expect(segment.duration > 0, "Duration should be positive")
        }
        
        print("VAD v5 detected \(segments.count) speech segments:")
        for (i, segment) in segments.enumerated() {
            print("  Segment \(i + 1): \(String(format: "%.2f", segment.startTime))s - \(String(format: "%.2f", segment.endTime))s (duration: \(String(format: "%.2f", segment.duration))s)")
        }
    }
    
    @Test("VAD v5 returns no speech for silence")
    func vadV5NoSpeechForSilence() async throws {
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV5ModelPath, useGPU: false)
        
        // Create 3 seconds of silence at 16kHz (longer duration for reliable detection)
        let silentSamples = [Float](repeating: 0.0, count: 48000)
        
        // For very short silence, VAD may still return true due to initialization,
        // so we check that probabilities are very low instead
        let probabilities = await vad.getSpeechProbabilities(samples: silentSamples)
        
        if !probabilities.isEmpty {
            let avgProb = probabilities.reduce(0, +) / Float(probabilities.count)
            let maxProb = probabilities.max() ?? 0
            print("VAD v5 silence - Avg probability: \(avgProb), Max: \(maxProb)")
            #expect(avgProb < 0.3, "Average speech probability for silence should be low")
        }
    }
    
    @Test("VAD v5 returns speech probabilities")
    func vadV5GetsProbabilities() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV5ModelPath, useGPU: false)
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        let probabilities = await vad.getSpeechProbabilities(samples: samples)
        
        #expect(!probabilities.isEmpty, "VAD v5 should return probabilities")
        
        // Probabilities should be between 0 and 1
        for prob in probabilities {
            #expect(prob >= 0.0 && prob <= 1.0, "Probability should be between 0 and 1, got \(prob)")
        }
        
        print("VAD v5 returned \(probabilities.count) probability values")
        print("  Min: \(String(format: "%.3f", probabilities.min() ?? 0))")
        print("  Max: \(String(format: "%.3f", probabilities.max() ?? 0))")
        print("  Avg: \(String(format: "%.3f", probabilities.reduce(0, +) / Float(probabilities.count)))")
    }
    
    // NOTE: StreamingTranscriber VAD integration tests are complex and require
    // real-time audio streaming simulation. They are tested separately in manual
    // integration tests with actual audio input devices.
}

// MARK: - VAD Tests (Silero v6)

@Suite("VAD v6 Tests", .enabled(if: TestFixtures.hasVADv6Model && TestFixtures.hasModel))
struct VADv6Tests {
    
    @Test("Can load Silero VAD v6 model")
    func loadVADv6Model() async throws {
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV6ModelPath, useGPU: false)
        _ = vad
    }
    
    @Test("VAD v6 detects speech in JFK audio")
    func vadV6DetectsSpeech() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV6ModelPath, useGPU: false)
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        let hasSpeech = await vad.detectSpeech(samples: samples)
        #expect(hasSpeech == true, "VAD v6 should detect speech in JFK audio")
    }
    
    @Test("VAD v6 returns speech segments")
    func vadV6GetsSpeechSegments() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV6ModelPath, useGPU: false)
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        let segments = try await vad.getSpeechSegments(samples: samples, options: .default)
        
        #expect(!segments.isEmpty, "VAD v6 should return speech segments")
        
        // Check segment timing makes sense
        for segment in segments {
            #expect(segment.startTime >= 0, "Start time should be non-negative")
            #expect(segment.endTime > segment.startTime, "End time should be after start time")
            #expect(segment.duration > 0, "Duration should be positive")
        }
        
        print("VAD v6 detected \(segments.count) speech segments:")
        for (i, segment) in segments.enumerated() {
            print("  Segment \(i + 1): \(String(format: "%.2f", segment.startTime))s - \(String(format: "%.2f", segment.endTime))s (duration: \(String(format: "%.2f", segment.duration))s)")
        }
    }
    
    @Test("VAD v6 returns no speech for silence")
    func vadV6NoSpeechForSilence() async throws {
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV6ModelPath, useGPU: false)
        
        // Create 3 seconds of silence at 16kHz (longer duration for reliable detection)
        let silentSamples = [Float](repeating: 0.0, count: 48000)
        
        // For very short silence, VAD may still return true due to initialization,
        // so we check that probabilities are very low instead
        let probabilities = await vad.getSpeechProbabilities(samples: silentSamples)
        
        if !probabilities.isEmpty {
            let avgProb = probabilities.reduce(0, +) / Float(probabilities.count)
            let maxProb = probabilities.max() ?? 0
            print("VAD v6 silence - Avg probability: \(avgProb), Max: \(maxProb)")
            #expect(avgProb < 0.3, "Average speech probability for silence should be low")
        }
    }
    
    @Test("VAD v6 returns speech probabilities")
    func vadV6GetsProbabilities() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Use CPU-only to avoid Metal backend issues with VAD
        let vad = try VADContext(modelPath: TestFixtures.vadV6ModelPath, useGPU: false)
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        let probabilities = await vad.getSpeechProbabilities(samples: samples)
        
        #expect(!probabilities.isEmpty, "VAD v6 should return probabilities")
        
        // Probabilities should be between 0 and 1
        for prob in probabilities {
            #expect(prob >= 0.0 && prob <= 1.0, "Probability should be between 0 and 1, got \(prob)")
        }
        
        print("VAD v6 returned \(probabilities.count) probability values")
        print("  Min: \(String(format: "%.3f", probabilities.min() ?? 0))")
        print("  Max: \(String(format: "%.3f", probabilities.max() ?? 0))")
        print("  Avg: \(String(format: "%.3f", probabilities.reduce(0, +) / Float(probabilities.count)))")
    }
    
    // NOTE: StreamingTranscriber VAD integration tests are complex and require
    // real-time audio streaming simulation. They are tested separately in manual
    // integration tests with actual audio input devices.
}

// MARK: - VAD Comparison Tests

@Suite("VAD Comparison Tests", .enabled(if: TestFixtures.hasVADv5Model && TestFixtures.hasVADv6Model && TestFixtures.hasModel))
struct VADComparisonTests {
    
    @Test("Both VAD versions detect speech in same audio")
    func bothVersionsDetectSpeech() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Use CPU-only to avoid Metal backend issues with VAD
        let vadV5 = try VADContext(modelPath: TestFixtures.vadV5ModelPath, useGPU: false)
        let vadV6 = try VADContext(modelPath: TestFixtures.vadV6ModelPath, useGPU: false)
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        let hasSpeechV5 = await vadV5.detectSpeech(samples: samples)
        let hasSpeechV6 = await vadV6.detectSpeech(samples: samples)
        
        #expect(hasSpeechV5 == true, "VAD v5 should detect speech")
        #expect(hasSpeechV6 == true, "VAD v6 should detect speech")
    }
    
    @Test("Both VAD versions return similar segment counts")
    func bothVersionsReturnSimilarSegments() async throws {
        guard TestFixtures.hasJFKAudio else {
            Issue.record("JFK audio not found")
            return
        }
        
        // Use CPU-only to avoid Metal backend issues with VAD
        let vadV5 = try VADContext(modelPath: TestFixtures.vadV5ModelPath, useGPU: false)
        let vadV6 = try VADContext(modelPath: TestFixtures.vadV6ModelPath, useGPU: false)
        let samples = try AudioProcessor.loadAudioFile(TestFixtures.jfkAudioPath)
        
        let segmentsV5 = try await vadV5.getSpeechSegments(samples: samples, options: .default)
        let segmentsV6 = try await vadV6.getSpeechSegments(samples: samples, options: .default)
        
        print("VAD v5 segments: \(segmentsV5.count)")
        print("VAD v6 segments: \(segmentsV6.count)")
        
        // Both should find at least one segment
        #expect(!segmentsV5.isEmpty, "VAD v5 should find segments")
        #expect(!segmentsV6.isEmpty, "VAD v6 should find segments")
        
        // Calculate total speech duration for each
        let totalDurationV5 = segmentsV5.reduce(0) { $0 + $1.duration }
        let totalDurationV6 = segmentsV6.reduce(0) { $0 + $1.duration }
        
        print("VAD v5 total speech duration: \(String(format: "%.2f", totalDurationV5))s")
        print("VAD v6 total speech duration: \(String(format: "%.2f", totalDurationV6))s")
        
        // Total durations should be somewhat similar (within 50% of each other)
        let ratio = max(totalDurationV5, totalDurationV6) / max(min(totalDurationV5, totalDurationV6), 0.1)
        #expect(ratio < 2.0, "VAD versions should have similar total speech durations (ratio: \(ratio))")
    }
}
