import Foundation
import whisper

/// A sendable wrapper around OpaquePointer for the whisper context.
/// This is safe because the actor ensures single-threaded access.
final class WhisperContextPointer: @unchecked Sendable {
    let pointer: OpaquePointer
    
    init(_ pointer: OpaquePointer) {
        self.pointer = pointer
    }
    
    deinit {
        whisper_free(pointer)
    }
}

/// Internal actor wrapping whisper_context for thread-safe access.
///
/// whisper.cpp's context is not thread-safe, so we use an actor to ensure
/// that only one operation accesses the context at a time.
actor WhisperContext {
    /// The underlying whisper context pointer wrapper.
    private let contextWrapper: WhisperContextPointer
    
    /// Direct access to the context pointer.
    private var context: OpaquePointer {
        contextWrapper.pointer
    }
    
    /// The configuration used to create this context.
    let configuration: WhisperConfiguration
    
    /// Creates a new whisper context from a model file.
    /// - Parameters:
    ///   - modelPath: Path to the GGML model file.
    ///   - configuration: Hardware configuration options.
    /// - Throws: `WhisperError.modelNotFound` or `WhisperError.modelLoadFailed`
    init(modelPath: URL, configuration: WhisperConfiguration) throws {
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw WhisperError.modelNotFound(modelPath)
        }
        
        var params = whisper_context_default_params()
        params.use_gpu = configuration.useGPU
        params.flash_attn = configuration.useFlashAttention
        
        guard let ctx = whisper_init_from_file_with_params(modelPath.path, params) else {
            throw WhisperError.modelLoadFailed(
                "Failed to initialize whisper context from \(modelPath.path)"
            )
        }
        
        self.contextWrapper = WhisperContextPointer(ctx)
        self.configuration = configuration
    }
    
    // MARK: - Transcription
    
    /// Minimum samples required for transcription (100ms at 16kHz).
    /// whisper.cpp requires at least 100ms of audio input.
    private static let minimumSamples = 1600
    
    /// Performs full transcription on audio samples.
    /// - Parameters:
    ///   - samples: Audio samples as Float32 values (-1.0 to 1.0), must be 16kHz mono.
    ///   - options: Transcription options.
    /// - Returns: Array of raw segments from the transcription.
    /// - Throws: `WhisperError.transcriptionFailed` on failure.
    func transcribe(
        samples: [Float],
        options: TranscriptionOptions
    ) throws -> [RawSegment] {
        // whisper.cpp requires at least 100ms of audio
        guard samples.count >= Self.minimumSamples else {
            // Return empty result for audio that's too short
            return []
        }
        
        var params = whisper_full_default_params(
            options.samplingStrategy.whisperStrategy
        )
        
        // Configure basic options
        params.n_threads = configuration.optimalThreadCount
        params.translate = options.translate
        params.print_realtime = false
        params.print_progress = false
        params.print_timestamps = false
        params.print_special = false
        
        // Language handling - for non-multilingual models (e.g., tiny.en),
        // we must set language to "en" and disable detection
        //
        // Note: detect_language must be false for transcription to work.
        // When detect_language is true, whisper.cpp only detects language and returns early.
        // To auto-detect language AND transcribe, set language to "auto" with detect_language = false.
        let languagePtr: UnsafeMutablePointer<CChar>?
        if !isMultilingual {
            // English-only model - always use English
            languagePtr = strdup("en")
            params.language = UnsafePointer(languagePtr)
            params.detect_language = false
        } else if let language = options.language, language != .auto {
            // Explicit language specified
            languagePtr = strdup(language.rawValue)
            params.language = UnsafePointer(languagePtr)
            params.detect_language = false
        } else {
            // Auto-detect language: set language to "auto" and detect_language to false
            // whisper.cpp will auto-detect when language is "auto" or nil, and continue transcription
            // when detect_language is false
            languagePtr = strdup("auto")
            params.language = UnsafePointer(languagePtr)
            params.detect_language = false
        }
        defer { languagePtr?.deallocate() }
        
        // Initial prompt
        let promptPtr: UnsafeMutablePointer<CChar>?
        if let prompt = options.initialPrompt {
            promptPtr = strdup(prompt)
            params.initial_prompt = UnsafePointer(promptPtr)
        } else {
            promptPtr = nil
        }
        defer { promptPtr?.deallocate() }
        
        // Token timestamps
        params.token_timestamps = options.tokenTimestamps
        
        // Beam search parameters
        if case .beamSearch(let beamSize) = options.samplingStrategy {
            params.beam_search.beam_size = Int32(beamSize)
        }
        
        // Run transcription
        whisper_reset_timings(context)
        
        // Call whisper_full - use helper to avoid closure capture issues with params
        let result = performTranscription(samples: samples, params: params)
        
        guard result == 0 else {
            throw WhisperError.transcriptionFailed(
                "whisper_full returned error code \(result)"
            )
        }
        
        return extractSegments()
    }
    
    /// Helper to perform the actual transcription call.
    /// Separated to avoid closure capture issues with params in Swift 6.
    private nonisolated func performTranscription(
        samples: [Float],
        params: whisper_full_params
    ) -> Int32 {
        samples.withUnsafeBufferPointer { samplesPtr in
            whisper_full(contextWrapper.pointer, params, samplesPtr.baseAddress, Int32(samples.count))
        }
    }
    
    // MARK: - Result Extraction
    
    /// Extracts all segments from the current context state.
    private func extractSegments() -> [RawSegment] {
        let segmentCount = whisper_full_n_segments(context)
        guard segmentCount > 0 else { return [] }
        
        return (0..<segmentCount).map { i in
            let text = String(cString: whisper_full_get_segment_text(context, i))
            let t0 = whisper_full_get_segment_t0(context, i)
            let t1 = whisper_full_get_segment_t1(context, i)
            let noSpeechProb = whisper_full_get_segment_no_speech_prob(context, i)
            let speakerTurn = whisper_full_get_segment_speaker_turn_next(context, i)
            
            // Extract tokens if available
            let tokenCount = whisper_full_n_tokens(context, i)
            var tokens: [RawToken]? = nil
            
            if tokenCount > 0 {
                tokens = (0..<tokenCount).map { j in
                    let tokenData = whisper_full_get_token_data(context, i, j)
                    let tokenText = String(cString: whisper_full_get_token_text(context, i, j))
                    return RawToken(
                        id: Int(tokenData.id),
                        text: tokenText,
                        probability: tokenData.p,
                        t0: tokenData.t0,
                        t1: tokenData.t1
                    )
                }
            }
            
            return RawSegment(
                index: Int(i),
                text: text,
                t0: t0,
                t1: t1,
                noSpeechProbability: noSpeechProb,
                isSpeakerTurn: speakerTurn,
                tokens: tokens
            )
        }
    }
    
    // MARK: - Model Information
    
    /// Returns whether the loaded model is multilingual.
    var isMultilingual: Bool {
        whisper_is_multilingual(context) != 0
    }
    
    /// Returns the detected language ID from the last transcription.
    var detectedLanguageId: Int32 {
        whisper_full_lang_id(context)
    }
    
    /// Returns the language string for a given language ID.
    func languageString(for id: Int32) -> String? {
        guard let ptr = whisper_lang_str(id) else { return nil }
        return String(cString: ptr)
    }
    
    // MARK: - Timings
    
    /// Returns timing information from the last transcription.
    func getTimings() -> RawTimings? {
        guard let timingsPtr = whisper_get_timings(context) else { return nil }
        let timings = timingsPtr.pointee
        return RawTimings(
            sampleMs: timings.sample_ms,
            encodeMs: timings.encode_ms,
            decodeMs: timings.decode_ms,
            batchdMs: timings.batchd_ms,
            promptMs: timings.prompt_ms
        )
    }
}

// MARK: - Raw Types

/// Raw segment data extracted from whisper.cpp.
struct RawSegment: Sendable {
    let index: Int
    let text: String
    /// Start time in centiseconds (1/100th of a second).
    let t0: Int64
    /// End time in centiseconds (1/100th of a second).
    let t1: Int64
    let noSpeechProbability: Float
    let isSpeakerTurn: Bool
    let tokens: [RawToken]?
    
    /// Start time in seconds.
    var startTime: TimeInterval {
        TimeInterval(t0) / 100.0
    }
    
    /// End time in seconds.
    var endTime: TimeInterval {
        TimeInterval(t1) / 100.0
    }
}

/// Raw token data extracted from whisper.cpp.
struct RawToken: Sendable {
    let id: Int
    let text: String
    let probability: Float
    /// Start time in centiseconds.
    let t0: Int64
    /// End time in centiseconds.
    let t1: Int64
    
    var startTime: TimeInterval? {
        t0 >= 0 ? TimeInterval(t0) / 100.0 : nil
    }
    
    var endTime: TimeInterval? {
        t1 >= 0 ? TimeInterval(t1) / 100.0 : nil
    }
}

/// Raw timing data from whisper.cpp.
struct RawTimings: Sendable {
    let sampleMs: Float
    let encodeMs: Float
    let decodeMs: Float
    let batchdMs: Float
    let promptMs: Float
    
    var totalMs: Float {
        sampleMs + encodeMs + decodeMs + batchdMs + promptMs
    }
}
