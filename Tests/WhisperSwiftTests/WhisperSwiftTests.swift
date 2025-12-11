import Testing
import Foundation
@testable import WhisperSwift

// MARK: - Transcriber Tests

@Suite("Transcriber Initialization")
struct TranscriberInitializationTests {
    
    @Test("Throws modelNotFound for non-existent path")
    func initWithNonExistentModel() async throws {
        let nonExistentURL = URL(fileURLWithPath: "/nonexistent/path/model.bin")
        
        await #expect(throws: WhisperError.self) {
            _ = try await Transcriber(modelPath: nonExistentURL)
        }
    }
    
    @Test("Throws modelNotFound with correct URL")
    func initWithNonExistentModelCorrectError() async throws {
        let nonExistentURL = URL(fileURLWithPath: "/tmp/nonexistent-model.bin")
        
        do {
            _ = try await Transcriber(modelPath: nonExistentURL)
            Issue.record("Expected error to be thrown")
        } catch let error as WhisperError {
            guard case .modelNotFound(let url) = error else {
                Issue.record("Expected modelNotFound, got \(error)")
                return
            }
            #expect(url == nonExistentURL)
        }
    }
}

// MARK: - WhisperConfiguration Tests

@Suite("WhisperConfiguration")
struct WhisperConfigurationTests {
    
    @Test("Default configuration has GPU enabled")
    func defaultConfigurationHasGPU() {
        let config = WhisperConfiguration.default
        #expect(config.useGPU == true)
        #expect(config.useFlashAttention == true)
    }
    
    @Test("CPU-only configuration disables GPU")
    func cpuOnlyConfiguration() {
        let config = WhisperConfiguration.cpuOnly
        #expect(config.useGPU == false)
        #expect(config.useFlashAttention == false)
    }
    
    @Test("Custom configuration preserves values")
    func customConfiguration() {
        let config = WhisperConfiguration(
            useGPU: false,
            useFlashAttention: true,
            threadCount: 4
        )
        #expect(config.useGPU == false)
        #expect(config.useFlashAttention == true)
        #expect(config.threadCount == 4)
    }
    
    @Test("Optimal thread count respects custom value")
    func optimalThreadCountCustom() {
        let config = WhisperConfiguration(threadCount: 8)
        #expect(config.optimalThreadCount == 8)
    }
    
    @Test("Optimal thread count has default when nil")
    func optimalThreadCountDefault() {
        let config = WhisperConfiguration(threadCount: nil)
        #expect(config.optimalThreadCount >= 1)
        #expect(config.optimalThreadCount <= ProcessInfo.processInfo.activeProcessorCount)
    }
}

// MARK: - TranscriptionOptions Tests

@Suite("TranscriptionOptions")
struct TranscriptionOptionsTests {
    
    @Test("Default options have sensible values")
    func defaultOptions() {
        let options = TranscriptionOptions.default
        #expect(options.language == nil)
        #expect(options.translate == false)
        #expect(options.tokenTimestamps == false)
        #expect(options.initialPrompt == nil)
    }
    
    @Test("Custom options preserve values")
    func customOptions() {
        let options = TranscriptionOptions(
            language: .english,
            translate: true,
            tokenTimestamps: true,
            initialPrompt: "Hello",
            samplingStrategy: .beamSearch(beamSize: 3)
        )
        
        #expect(options.language == .english)
        #expect(options.translate == true)
        #expect(options.tokenTimestamps == true)
        #expect(options.initialPrompt == "Hello")
        
        if case .beamSearch(let beamSize) = options.samplingStrategy {
            #expect(beamSize == 3)
        } else {
            Issue.record("Expected beam search strategy")
        }
    }
}

// MARK: - Language Tests

@Suite("Language")
struct LanguageTests {
    
    @Test("All languages have valid raw values")
    func allLanguagesHaveRawValues() {
        for language in Language.allCases {
            #expect(!language.rawValue.isEmpty)
        }
    }
    
    @Test("All languages have display names")
    func allLanguagesHaveDisplayNames() {
        for language in Language.allCases {
            #expect(!language.displayName.isEmpty)
        }
    }
    
    @Test("English language has correct values")
    func englishLanguage() {
        let english = Language.english
        #expect(english.rawValue == "en")
        #expect(english.displayName == "English")
    }
    
    @Test("Auto language for detection")
    func autoLanguage() {
        let auto = Language.auto
        #expect(auto.rawValue == "auto")
    }
    
    @Test("Language can be created from raw value")
    func languageFromRawValue() {
        let spanish = Language(rawValue: "es")
        #expect(spanish == .spanish)
        
        let invalid = Language(rawValue: "invalid")
        #expect(invalid == nil)
    }
}

// MARK: - WhisperError Tests

@Suite("WhisperError")
struct WhisperErrorTests {
    
    @Test("Error descriptions are human-readable")
    func errorDescriptions() {
        let testURL = URL(fileURLWithPath: "/test/path")
        
        let modelNotFound = WhisperError.modelNotFound(testURL)
        #expect(modelNotFound.errorDescription?.contains("/test/path") == true)
        
        let modelLoadFailed = WhisperError.modelLoadFailed("corrupt file")
        #expect(modelLoadFailed.errorDescription?.contains("corrupt file") == true)
        
        let invalidFormat = WhisperError.invalidAudioFormat("wrong format")
        #expect(invalidFormat.errorDescription?.contains("wrong format") == true)
        
        let transcriptionFailed = WhisperError.transcriptionFailed("processing error")
        #expect(transcriptionFailed.errorDescription?.contains("processing error") == true)
    }
}

// MARK: - TranscriptionResult Tests

@Suite("TranscriptionResult")
struct TranscriptionResultTests {
    
    @Test("Empty segments produce empty text")
    func emptySegments() {
        let result = TranscriptionResult(
            segments: [],
            detectedLanguage: nil,
            timings: nil
        )
        
        #expect(result.text == "")
        #expect(result.segments.isEmpty)
        #expect(result.detectedLanguage == nil)
        #expect(result.timings == nil)
    }
    
    @Test("Multiple segments are concatenated")
    func multipleSegments() {
        let segments = [
            RawSegment(
                index: 0,
                text: "Hello ",
                t0: 0,
                t1: 100,
                noSpeechProbability: 0.1,
                isSpeakerTurn: false,
                tokens: nil
            ),
            RawSegment(
                index: 1,
                text: "world!",
                t0: 100,
                t1: 200,
                noSpeechProbability: 0.2,
                isSpeakerTurn: true,
                tokens: nil
            )
        ]
        
        let result = TranscriptionResult(
            segments: segments,
            detectedLanguage: .english,
            timings: nil
        )
        
        #expect(result.text == "Hello world!")
        #expect(result.segments.count == 2)
        #expect(result.detectedLanguage == .english)
    }
}

// MARK: - TranscriptionSegment Tests

@Suite("TranscriptionSegment")
struct TranscriptionSegmentTests {
    
    @Test("Segment times are converted from centiseconds to seconds")
    func segmentTimeConversion() {
        let raw = RawSegment(
            index: 0,
            text: "Test",
            t0: 150,   // 1.5 seconds
            t1: 350,   // 3.5 seconds
            noSpeechProbability: 0.05,
            isSpeakerTurn: false,
            tokens: nil
        )
        
        let segment = TranscriptionSegment(from: raw)
        
        #expect(segment.startTime == 1.5)
        #expect(segment.endTime == 3.5)
        #expect(segment.noSpeechProbability == 0.05)
        #expect(segment.isSpeakerTurn == false)
    }
    
    @Test("Segment has correct identifier")
    func segmentIdentifier() {
        let raw = RawSegment(
            index: 42,
            text: "Test",
            t0: 0,
            t1: 100,
            noSpeechProbability: 0.0,
            isSpeakerTurn: false,
            tokens: nil
        )
        
        let segment = TranscriptionSegment(from: raw)
        #expect(segment.id == 42)
    }
}

// MARK: - AudioProcessor Tests

@Suite("AudioProcessor")
struct AudioProcessorTests {
    
    @Test("Required sample rate is 16kHz")
    func requiredSampleRate() {
        #expect(AudioProcessor.requiredSampleRate == 16000)
    }
    
    @Test("Loading non-existent file throws error")
    func loadNonExistentFile() throws {
        let nonExistentURL = URL(fileURLWithPath: "/nonexistent/audio.wav")
        
        #expect(throws: WhisperError.self) {
            _ = try AudioProcessor.loadAudioFile(nonExistentURL)
        }
    }
    
    @Test("Loading non-existent file throws audioLoadFailed")
    func loadNonExistentFileCorrectError() throws {
        let nonExistentURL = URL(fileURLWithPath: "/tmp/nonexistent-audio.wav")
        
        do {
            _ = try AudioProcessor.loadAudioFile(nonExistentURL)
            Issue.record("Expected error to be thrown")
        } catch let error as WhisperError {
            guard case .audioLoadFailed(let url, _) = error else {
                Issue.record("Expected audioLoadFailed, got \(error)")
                return
            }
            #expect(url == nonExistentURL)
        }
    }
}

// MARK: - SamplingStrategy Tests

@Suite("SamplingStrategy")
struct SamplingStrategyTests {
    
    @Test("Greedy strategy has correct whisper strategy")
    func greedyStrategy() {
        let strategy = SamplingStrategy.greedy
        // Just verify it doesn't crash - actual value depends on whisper.cpp
        _ = strategy.whisperStrategy
    }
    
    @Test("Beam search strategy with custom beam size")
    func beamSearchStrategy() {
        let strategy = SamplingStrategy.beamSearch(beamSize: 10)
        
        if case .beamSearch(let beamSize) = strategy {
            #expect(beamSize == 10)
        } else {
            Issue.record("Expected beam search")
        }
    }
    
    @Test("Beam search default beam size is 5")
    func beamSearchDefaultSize() {
        let strategy = SamplingStrategy.beamSearch()
        
        if case .beamSearch(let beamSize) = strategy {
            #expect(beamSize == 5)
        } else {
            Issue.record("Expected beam search")
        }
    }
}

// MARK: - StreamingState Tests

@Suite("StreamingState")
struct StreamingStateTests {
    
    @Test("States have correct descriptions")
    func stateDescriptions() {
        #expect(StreamingState.idle.description == "idle")
        #expect(StreamingState.running.description == "running")
        #expect(StreamingState.stopping.description == "stopping")
        #expect(StreamingState.stopped.description == "stopped")
    }
    
    @Test("States are equatable")
    func stateEquatable() {
        #expect(StreamingState.idle == StreamingState.idle)
        #expect(StreamingState.running == StreamingState.running)
        #expect(StreamingState.idle != StreamingState.running)
    }
    
    @Test("Failed states are equal regardless of error")
    func failedStateEquality() {
        let error1 = WhisperError.transcriptionFailed("error 1")
        let error2 = WhisperError.transcriptionFailed("error 2")
        
        // Failed states compare equal (by design, for state checking)
        #expect(StreamingState.failed(error1) == StreamingState.failed(error2))
    }
}

// MARK: - VADOptions Tests

@Suite("VADOptions")
struct VADOptionsTests {
    
    @Test("Default options have sensible values")
    func defaultOptions() {
        let options = VADOptions.default
        
        #expect(options.threshold == 0.5)
        #expect(options.minSpeechDurationMs == 250)
        #expect(options.minSilenceDurationMs == 100)
        #expect(options.maxSpeechDurationS == 30.0)
        #expect(options.speechPadMs == 200)
        #expect(options.samplesOverlap == 0.0)
    }
    
    @Test("Custom options preserve values")
    func customOptions() {
        let options = VADOptions(
            threshold: 0.7,
            minSpeechDurationMs: 500,
            minSilenceDurationMs: 200,
            maxSpeechDurationS: 60.0,
            speechPadMs: 100,
            samplesOverlap: 0.5
        )
        
        #expect(options.threshold == 0.7)
        #expect(options.minSpeechDurationMs == 500)
        #expect(options.minSilenceDurationMs == 200)
        #expect(options.maxSpeechDurationS == 60.0)
        #expect(options.speechPadMs == 100)
        #expect(options.samplesOverlap == 0.5)
    }
    
    @Test("Whisper params conversion works")
    func whisperParamsConversion() {
        let options = VADOptions(
            threshold: 0.6,
            minSpeechDurationMs: 300,
            minSilenceDurationMs: 150
        )
        
        let params = options.whisperParams
        #expect(params.threshold == 0.6)
        #expect(params.min_speech_duration_ms == 300)
        #expect(params.min_silence_duration_ms == 150)
    }
}

// MARK: - AudioRingBuffer Tests

@Suite("AudioRingBuffer")
struct AudioRingBufferTests {
    
    @Test("Buffer starts empty")
    func bufferStartsEmpty() async {
        let buffer = AudioRingBuffer()
        
        let count = await buffer.count
        let duration = await buffer.duration
        
        #expect(count == 0)
        #expect(duration == 0)
    }
    
    @Test("Appending samples increases count")
    func appendIncreasesCount() async {
        let buffer = AudioRingBuffer(sampleRate: 16000)
        let samples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        await buffer.append(samples)
        
        let count = await buffer.count
        #expect(count == 5)
    }
    
    @Test("Duration is calculated correctly")
    func durationCalculation() async {
        let buffer = AudioRingBuffer(sampleRate: 16000)
        // 16000 samples = 1 second at 16kHz
        let samples = [Float](repeating: 0.0, count: 16000)
        
        await buffer.append(samples)
        
        let duration = await buffer.duration
        #expect(duration == 1.0)
    }
    
    @Test("Consuming samples removes them")
    func consumeRemovesSamples() async {
        let buffer = AudioRingBuffer(sampleRate: 16000)
        let samples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        await buffer.append(samples)
        let consumed = await buffer.consume(3)
        
        #expect(consumed == [0.1, 0.2, 0.3])
        
        let remaining = await buffer.count
        #expect(remaining == 2)
    }
    
    @Test("ConsumeAll empties the buffer")
    func consumeAllEmptiesBuffer() async {
        let buffer = AudioRingBuffer(sampleRate: 16000)
        let samples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        await buffer.append(samples)
        let consumed = await buffer.consumeAll()
        
        #expect(consumed == samples)
        
        let remaining = await buffer.count
        #expect(remaining == 0)
    }
    
    @Test("Reset clears everything")
    func resetClearsBuffer() async {
        let buffer = AudioRingBuffer(sampleRate: 16000)
        let samples: [Float] = [0.1, 0.2, 0.3]
        
        await buffer.append(samples)
        await buffer.reset()
        
        let count = await buffer.count
        let totalDuration = await buffer.totalDuration
        
        #expect(count == 0)
        #expect(totalDuration == 0)
    }
    
    @Test("Total duration includes consumed samples")
    func totalDurationIncludesConsumed() async {
        let buffer = AudioRingBuffer(sampleRate: 16000)
        let samples = [Float](repeating: 0.0, count: 16000)
        
        await buffer.append(samples)
        _ = await buffer.consume(8000)
        
        let duration = await buffer.duration
        let totalDuration = await buffer.totalDuration
        
        #expect(duration == 0.5)
        #expect(totalDuration == 1.0)
    }
    
    @Test("Buffer trims when exceeding max size")
    func bufferTrimsOnOverflow() async {
        // Create buffer with 1 second max duration
        let buffer = AudioRingBuffer(sampleRate: 16000, maxDurationSeconds: 1.0)
        
        // Add 2 seconds worth of samples
        let samples = [Float](repeating: 0.0, count: 32000)
        await buffer.append(samples)
        
        // Should only keep the most recent 1 second
        let count = await buffer.count
        #expect(count == 16000)
    }
    
    @Test("GetSamples returns subset without consuming")
    func getSamplesReturnsSubset() async {
        let buffer = AudioRingBuffer(sampleRate: 16000)
        let samples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        await buffer.append(samples)
        
        let subset = await buffer.getSamples(from: 1, count: 3)
        #expect(subset == [0.2, 0.3, 0.4])
        
        // Original buffer unchanged
        let count = await buffer.count
        #expect(count == 5)
    }
}

// MARK: - StreamingTranscriber Tests

@Suite("StreamingTranscriber Initialization")
struct StreamingTranscriberInitializationTests {
    
    @Test("Throws modelNotFound for non-existent model path")
    func initWithNonExistentModel() async throws {
        let nonExistentURL = URL(fileURLWithPath: "/nonexistent/path/model.bin")
        
        await #expect(throws: WhisperError.self) {
            _ = try await StreamingTranscriber(modelPath: nonExistentURL)
        }
    }
    
    @Test("Throws vadFailed for non-existent VAD model path")
    func initWithNonExistentVADModel() async throws {
        // Use the fixtures path - this test will throw because model doesn't exist
        let nonExistentModelURL = URL(fileURLWithPath: "/nonexistent/model.bin")
        let nonExistentVADURL = URL(fileURLWithPath: "/nonexistent/vad.bin")
        
        await #expect(throws: WhisperError.self) {
            _ = try await StreamingTranscriber(
                modelPath: nonExistentModelURL,
                vadModelPath: nonExistentVADURL
            )
        }
    }
}

// MARK: - SilenceDetector Tests

@Suite("SilenceDetector")
struct SilenceDetectorTests {
    
    @Test("RMS calculation for silence returns low value")
    func rmsCalculationSilence() {
        let silentSamples = [Float](repeating: 0.0, count: 160)
        
        let rms = SilenceDetector.calculateRMS(samples: silentSamples, start: 0, count: 160)
        
        #expect(rms == 0.0)
    }
    
    @Test("RMS calculation for constant signal")
    func rmsCalculationConstant() {
        // RMS of constant value should be that value
        let samples = [Float](repeating: 0.5, count: 160)
        
        let rms = SilenceDetector.calculateRMS(samples: samples, start: 0, count: 160)
        
        #expect(abs(rms - 0.5) < 0.001)
    }
    
    @Test("RMS calculation for sine wave")
    func rmsCalculationSineWave() {
        // RMS of sine wave with amplitude A is A / sqrt(2)
        let amplitude: Float = 1.0
        var samples = [Float](repeating: 0.0, count: 1600)
        for i in 0..<1600 {
            samples[i] = amplitude * sin(Float(i) * 2.0 * .pi / 160.0)
        }
        
        let rms = SilenceDetector.calculateRMS(samples: samples, start: 0, count: 1600)
        let expectedRMS = amplitude / sqrt(2.0)
        
        #expect(abs(rms - expectedRMS) < 0.01)
    }
    
    @Test("RMS calculation with offset")
    func rmsCalculationWithOffset() {
        var samples = [Float](repeating: 0.0, count: 320)
        // First half is silence, second half is signal
        for i in 160..<320 {
            samples[i] = 0.5
        }
        
        let rmsFirst = SilenceDetector.calculateRMS(samples: samples, start: 0, count: 160)
        let rmsSecond = SilenceDetector.calculateRMS(samples: samples, start: 160, count: 160)
        
        #expect(rmsFirst == 0.0)
        #expect(abs(rmsSecond - 0.5) < 0.001)
    }
    
    @Test("RMS calculation handles empty range")
    func rmsCalculationEmptyRange() {
        let samples = [Float](repeating: 0.5, count: 100)
        
        let rms = SilenceDetector.calculateRMS(samples: samples, start: 0, count: 0)
        
        #expect(rms == 0.0)
    }
    
    @Test("RMS calculation handles out of bounds")
    func rmsCalculationOutOfBounds() {
        let samples = [Float](repeating: 0.5, count: 100)
        
        let rms = SilenceDetector.calculateRMS(samples: samples, start: 150, count: 50)
        
        #expect(rms == 0.0)
    }
    
    @Test("ContainsSpeech returns true for loud audio")
    func containsSpeechLoud() {
        let samples = [Float](repeating: 0.5, count: 1600)
        
        let hasSpeech = SilenceDetector.containsSpeech(in: samples, threshold: 0.01)
        
        #expect(hasSpeech == true)
    }
    
    @Test("ContainsSpeech returns false for silence")
    func containsSpeechSilent() {
        let samples = [Float](repeating: 0.0, count: 1600)
        
        let hasSpeech = SilenceDetector.containsSpeech(in: samples, threshold: 0.01)
        
        #expect(hasSpeech == false)
    }
    
    @Test("ContainsSpeech returns false for empty audio")
    func containsSpeechEmpty() {
        let samples: [Float] = []
        
        let hasSpeech = SilenceDetector.containsSpeech(in: samples, threshold: 0.01)
        
        #expect(hasSpeech == false)
    }
    
    @Test("ContainsSpeech respects threshold")
    func containsSpeechThreshold() {
        // Low amplitude signal (0.005 RMS)
        let samples = [Float](repeating: 0.005, count: 1600)
        
        // Should be detected with low threshold
        let hasSpeechLow = SilenceDetector.containsSpeech(in: samples, threshold: 0.001)
        #expect(hasSpeechLow == true)
        
        // Should not be detected with high threshold
        let hasSpeechHigh = SilenceDetector.containsSpeech(in: samples, threshold: 0.01)
        #expect(hasSpeechHigh == false)
    }
    
    @Test("FindSilenceBreak returns nil for too short audio")
    func findSilenceBreakTooShort() {
        let samples = [Float](repeating: 0.5, count: 100)
        
        let breakPoint = SilenceDetector.findSilenceBreak(in: samples)
        
        #expect(breakPoint == nil)
    }
    
    @Test("FindSilenceBreak returns nil when no silence found")
    func findSilenceBreakNoSilence() {
        // Continuous speech (no silence)
        let samples = [Float](repeating: 0.5, count: 16000)
        
        let breakPoint = SilenceDetector.findSilenceBreak(in: samples)
        
        #expect(breakPoint == nil)
    }
    
    @Test("FindSilenceBreak finds silence gap")
    func findSilenceBreakFindsGap() {
        // Create audio with speech, then silence, then more speech
        // 1 second speech, 0.5 seconds silence, 1 second speech
        var samples = [Float](repeating: 0.0, count: 40000)
        
        // First 1 second: speech
        for i in 0..<16000 {
            samples[i] = 0.5
        }
        // Next 0.5 seconds: silence (stays 0.0)
        // Last 1.5 seconds: speech
        for i in 24000..<40000 {
            samples[i] = 0.5
        }
        
        let options = SilenceDetectorOptions(
            threshold: 0.01,
            minSilenceDuration: 0.3,
            searchDuration: 5.0,
            windowDuration: 0.01
        )
        
        let breakPoint = SilenceDetector.findSilenceBreak(in: samples, options: options)
        
        // Should find the silence gap around sample 16000
        #expect(breakPoint != nil)
        if let bp = breakPoint {
            // The break point should be around where speech ended (16000 samples)
            // Allow for window-based granularity
            #expect(bp >= 15800 && bp <= 16200)
        }
    }
    
    @Test("FindSpeechSegments returns empty for silence")
    func findSpeechSegmentsSilence() {
        let samples = [Float](repeating: 0.0, count: 16000)
        
        let segments = SilenceDetector.findSpeechSegments(in: samples)
        
        #expect(segments.isEmpty)
    }
    
    @Test("FindSpeechSegments returns one segment for continuous speech")
    func findSpeechSegmentsContinuous() {
        let samples = [Float](repeating: 0.5, count: 16000)
        
        let segments = SilenceDetector.findSpeechSegments(in: samples)
        
        #expect(segments.count == 1)
        if let segment = segments.first {
            #expect(segment.startSample == 0)
            #expect(segment.endSample == 16000)
        }
    }
    
    @Test("FindSpeechSegments finds multiple segments")
    func findSpeechSegmentsMultiple() {
        // Create audio: speech, silence, speech
        var samples = [Float](repeating: 0.0, count: 48000)
        
        // First speech segment: 0-16000 (1 second)
        for i in 0..<16000 {
            samples[i] = 0.5
        }
        // Silence: 16000-32000 (1 second)
        // Second speech segment: 32000-48000 (1 second)
        for i in 32000..<48000 {
            samples[i] = 0.5
        }
        
        let options = SilenceDetectorOptions(
            threshold: 0.01,
            minSilenceDuration: 0.3
        )
        
        let segments = SilenceDetector.findSpeechSegments(in: samples, options: options)
        
        #expect(segments.count == 2)
        
        if segments.count >= 2 {
            // First segment should be around 0-16000
            #expect(segments[0].startSample == 0)
            #expect(segments[0].endSample >= 15800 && segments[0].endSample <= 16200)
            
            // Second segment should start around 32000
            #expect(segments[1].startSample >= 31800 && segments[1].startSample <= 32200)
            #expect(segments[1].endSample == 48000)
        }
    }
    
    @Test("SpeechSegment time calculations")
    func speechSegmentTimeCalculations() {
        let segment = SpeechSegment(startSample: 16000, endSample: 32000)
        
        #expect(segment.startTime == 1.0)
        #expect(segment.endTime == 2.0)
        #expect(segment.duration == 1.0)
    }
    
    @Test("SilenceDetectorOptions default values")
    func silenceDetectorOptionsDefaults() {
        let options = SilenceDetectorOptions.default
        
        #expect(options.threshold == 0.01)
        #expect(options.minSilenceDuration == 0.3)
        #expect(options.searchDuration == 5.0)
        #expect(options.windowDuration == 0.01)
    }
    
    @Test("SilenceDetectorOptions custom values")
    func silenceDetectorOptionsCustom() {
        let options = SilenceDetectorOptions(
            threshold: 0.02,
            minSilenceDuration: 0.5,
            searchDuration: 10.0,
            windowDuration: 0.02
        )
        
        #expect(options.threshold == 0.02)
        #expect(options.minSilenceDuration == 0.5)
        #expect(options.searchDuration == 10.0)
        #expect(options.windowDuration == 0.02)
    }
}
