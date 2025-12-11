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
    
    /// Creates transcription options with the specified parameters.
    /// - Parameters:
    ///   - language: The language of the audio. Defaults to `nil` (auto-detect).
    ///   - translate: Whether to translate to English. Defaults to `false`.
    ///   - tokenTimestamps: Whether to include token timestamps. Defaults to `false`.
    ///   - initialPrompt: Initial prompt for conditioning. Defaults to `nil`.
    ///   - samplingStrategy: Decoding strategy. Defaults to `.greedy`.
    public init(
        language: Language? = nil,
        translate: Bool = false,
        tokenTimestamps: Bool = false,
        initialPrompt: String? = nil,
        samplingStrategy: SamplingStrategy = .greedy
    ) {
        self.language = language
        self.translate = translate
        self.tokenTimestamps = tokenTimestamps
        self.initialPrompt = initialPrompt
        self.samplingStrategy = samplingStrategy
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
