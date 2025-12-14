import Foundation
import whisper

/// Options for transcription behavior.
public struct TranscriptionOptions: Sendable {
    /// The language of the audio.
    ///
    /// Set to `nil` or `.auto` for automatic language detection.
    public var language: Language?
    
    /// Whether to translate to English instead of transcribing.
    public var translate: Bool
    
    /// Whether to include token-level timestamps.
    public var tokenTimestamps: Bool
    
    /// Initial prompt to condition the model.
    ///
    /// Can be used to provide context or guide transcription style.
    public var initialPrompt: String?
    
    /// Sampling strategy for decoding.
    public var samplingStrategy: SamplingStrategy
    
    /// Force single segment output (useful for streaming).
    ///
    /// When true, the transcription will return at most one segment
    /// regardless of audio length. Useful for real-time streaming.
    public var singleSegment: Bool
    
    /// Maximum number of tokens per audio chunk.
    ///
    /// Set to 0 for no limit. Lower values can speed up transcription
    /// but may truncate output.
    public var maxTokens: Int
    
    /// Whether to print timestamps in the output.
    ///
    /// When false, timestamps are not included in segment output.
    public var noTimestamps: Bool
    
    /// Creates transcription options with the specified parameters.
    /// - Parameters:
    ///   - language: The language of the audio. Defaults to `nil` (auto-detect).
    ///   - translate: Whether to translate to English. Defaults to `false`.
    ///   - tokenTimestamps: Whether to include token timestamps. Defaults to `false`.
    ///   - initialPrompt: Initial prompt for conditioning. Defaults to `nil`.
    ///   - samplingStrategy: Decoding strategy. Defaults to `.greedy`.
    ///   - singleSegment: Force single segment output. Defaults to `false`.
    ///   - maxTokens: Maximum tokens per chunk. Defaults to `0` (no limit).
    ///   - noTimestamps: Disable timestamps. Defaults to `false`.
    public init(
        language: Language? = nil,
        translate: Bool = false,
        tokenTimestamps: Bool = false,
        initialPrompt: String? = nil,
        samplingStrategy: SamplingStrategy = .greedy,
        singleSegment: Bool = false,
        maxTokens: Int = 0,
        noTimestamps: Bool = false
    ) {
        self.language = language
        self.translate = translate
        self.tokenTimestamps = tokenTimestamps
        self.initialPrompt = initialPrompt
        self.samplingStrategy = samplingStrategy
        self.singleSegment = singleSegment
        self.maxTokens = maxTokens
        self.noTimestamps = noTimestamps
    }
    
    /// Default options for general transcription.
    public static let `default` = TranscriptionOptions()
}

/// Sampling strategy for the decoder.
public enum SamplingStrategy: Sendable {
    /// Greedy decoding - fastest, good for most cases.
    case greedy
    
    /// Beam search - slower but potentially more accurate.
    case beamSearch(beamSize: Int = 5)
    
    /// Returns the whisper.cpp sampling strategy enum.
    internal var whisperStrategy: whisper_sampling_strategy {
        switch self {
        case .greedy:
            return WHISPER_SAMPLING_GREEDY
        case .beamSearch:
            return WHISPER_SAMPLING_BEAM_SEARCH
        }
    }
}
