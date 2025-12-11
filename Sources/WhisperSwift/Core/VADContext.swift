import Foundation
import whisper

/// A sendable wrapper around OpaquePointer for the VAD context.
/// This is safe because the actor ensures single-threaded access.
final class VADContextPointer: @unchecked Sendable {
    let pointer: OpaquePointer
    
    init(_ pointer: OpaquePointer) {
        self.pointer = pointer
    }
    
    deinit {
        whisper_vad_free(pointer)
    }
}

/// A detected speech segment from VAD processing.
struct VADSpeechSegment: Sendable {
    /// Start time in seconds.
    let startTime: Float
    /// End time in seconds.
    let endTime: Float
    
    /// Duration in seconds.
    var duration: Float {
        endTime - startTime
    }
}

/// Internal actor wrapping whisper_vad_context for thread-safe access.
///
/// This actor provides Voice Activity Detection using the Silero VAD model
/// integrated into whisper.cpp.
actor VADContext {
    /// The underlying VAD context pointer wrapper.
    private let contextWrapper: VADContextPointer
    
    /// Direct access to the context pointer.
    private var context: OpaquePointer {
        contextWrapper.pointer
    }
    
    /// Creates a new VAD context from a model file.
    /// - Parameters:
    ///   - modelPath: Path to the Silero VAD GGML model file.
    ///   - useGPU: Whether to use GPU acceleration. Defaults to true.
    ///   - threadCount: Number of threads to use. Defaults to 4.
    /// - Throws: `WhisperError.vadFailed` if the model cannot be loaded.
    init(modelPath: URL, useGPU: Bool = true, threadCount: Int = 4) throws {
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw WhisperError.vadFailed("VAD model not found at: \(modelPath.path)")
        }
        
        var params = whisper_vad_default_context_params()
        params.n_threads = Int32(threadCount)
        params.use_gpu = useGPU
        
        guard let ctx = whisper_vad_init_from_file_with_params(modelPath.path, params) else {
            throw WhisperError.vadFailed("Failed to initialize VAD context from \(modelPath.path)")
        }
        
        self.contextWrapper = VADContextPointer(ctx)
    }
    
    // MARK: - VAD Processing
    
    /// Detects whether speech is present in the audio samples.
    /// - Parameter samples: Audio samples at 16kHz mono.
    /// - Returns: `true` if speech is detected.
    func detectSpeech(samples: [Float]) -> Bool {
        samples.withUnsafeBufferPointer { ptr in
            whisper_vad_detect_speech(context, ptr.baseAddress, Int32(samples.count))
        }
    }
    
    /// Gets speech segments from audio samples using the specified options.
    /// - Parameters:
    ///   - samples: Audio samples at 16kHz mono.
    ///   - options: VAD processing options.
    /// - Returns: Array of detected speech segments.
    /// - Throws: `WhisperError.vadFailed` if processing fails.
    func getSpeechSegments(samples: [Float], options: VADOptions) throws -> [VADSpeechSegment] {
        let vadParams = options.whisperParams
        
        let segmentsPtr: OpaquePointer? = samples.withUnsafeBufferPointer { ptr in
            whisper_vad_segments_from_samples(
                context,
                vadParams,
                ptr.baseAddress,
                Int32(samples.count)
            )
        }
        
        guard let segments = segmentsPtr else {
            throw WhisperError.vadFailed("Failed to detect speech segments")
        }
        
        defer { whisper_vad_free_segments(segments) }
        
        let count = whisper_vad_segments_n_segments(segments)
        guard count > 0 else { return [] }
        
        return (0..<count).map { i in
            VADSpeechSegment(
                startTime: whisper_vad_segments_get_segment_t0(segments, i),
                endTime: whisper_vad_segments_get_segment_t1(segments, i)
            )
        }
    }
    
    /// Gets speech probabilities from audio samples.
    /// - Parameter samples: Audio samples at 16kHz mono.
    /// - Returns: Array of probability values for each audio chunk.
    func getSpeechProbabilities(samples: [Float]) -> [Float] {
        let hasSpeech = samples.withUnsafeBufferPointer { ptr in
            whisper_vad_detect_speech(context, ptr.baseAddress, Int32(samples.count))
        }
        
        guard hasSpeech else { return [] }
        
        let probCount = whisper_vad_n_probs(context)
        guard probCount > 0 else { return [] }
        
        guard let probsPtr = whisper_vad_probs(context) else { return [] }
        
        return Array(UnsafeBufferPointer(start: probsPtr, count: Int(probCount)))
    }
}
