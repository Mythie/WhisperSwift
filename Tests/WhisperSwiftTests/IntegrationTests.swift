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
