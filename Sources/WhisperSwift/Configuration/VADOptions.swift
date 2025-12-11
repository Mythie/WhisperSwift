import Foundation
import whisper

/// Voice Activity Detection options.
///
/// These options control how speech segments are detected in the audio stream.
/// VAD is used by ``StreamingTranscriber`` to determine when speech starts
/// and ends, enabling efficient real-time transcription.
public struct VADOptions: Sendable {
    /// Probability threshold to consider audio as speech (0.0-1.0).
    ///
    /// Higher values require more confidence that audio contains speech.
    /// - Default: 0.5
    public var threshold: Float
    
    /// Minimum duration for a valid speech segment in milliseconds.
    ///
    /// Speech segments shorter than this will be ignored.
    /// - Default: 250ms
    public var minSpeechDurationMs: Int
    
    /// Minimum silence duration to consider speech as ended in milliseconds.
    ///
    /// After this duration of silence, the current speech segment is finalized.
    /// - Default: 100ms
    public var minSilenceDurationMs: Int
    
    /// Maximum duration of a speech segment in seconds before forcing a new segment.
    ///
    /// Long speech segments are split at this duration.
    /// - Default: 30.0 seconds (matches whisper chunk size)
    public var maxSpeechDurationS: Float
    
    /// Padding added before and after detected speech segments in milliseconds.
    ///
    /// This helps ensure speech boundaries aren't clipped too aggressively.
    /// - Default: 200ms
    public var speechPadMs: Int
    
    /// Overlap in seconds when copying audio samples from speech segment.
    ///
    /// - Default: 0.0
    public var samplesOverlap: Float
    
    /// Creates VAD options with the specified parameters.
    /// - Parameters:
    ///   - threshold: Speech probability threshold. Defaults to 0.5.
    ///   - minSpeechDurationMs: Minimum speech duration. Defaults to 250ms.
    ///   - minSilenceDurationMs: Minimum silence duration. Defaults to 100ms.
    ///   - maxSpeechDurationS: Maximum speech duration. Defaults to 30.0s.
    ///   - speechPadMs: Speech padding. Defaults to 200ms.
    ///   - samplesOverlap: Samples overlap. Defaults to 0.0.
    public init(
        threshold: Float = 0.5,
        minSpeechDurationMs: Int = 250,
        minSilenceDurationMs: Int = 100,
        maxSpeechDurationS: Float = 30.0,
        speechPadMs: Int = 200,
        samplesOverlap: Float = 0.0
    ) {
        self.threshold = threshold
        self.minSpeechDurationMs = minSpeechDurationMs
        self.minSilenceDurationMs = minSilenceDurationMs
        self.maxSpeechDurationS = maxSpeechDurationS
        self.speechPadMs = speechPadMs
        self.samplesOverlap = samplesOverlap
    }
    
    /// Default VAD options.
    public static let `default` = VADOptions()
    
    /// Converts to whisper.cpp VAD parameters.
    internal var whisperParams: whisper_vad_params {
        var params = whisper_vad_default_params()
        params.threshold = threshold
        params.min_speech_duration_ms = Int32(minSpeechDurationMs)
        params.min_silence_duration_ms = Int32(minSilenceDurationMs)
        params.max_speech_duration_s = maxSpeechDurationS
        params.speech_pad_ms = Int32(speechPadMs)
        params.samples_overlap = samplesOverlap
        return params
    }
}
