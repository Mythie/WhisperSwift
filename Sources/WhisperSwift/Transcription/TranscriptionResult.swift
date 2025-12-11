import Foundation

/// The result of a transcription operation.
public struct TranscriptionResult: Sendable {
    /// The complete transcribed text.
    public let text: String
    
    /// Individual segments with timing information.
    public let segments: [TranscriptionSegment]
    
    /// The detected language, if auto-detection was used.
    public let detectedLanguage: Language?
    
    /// Performance timing information.
    public let timings: TranscriptionTimings?
    
    /// Creates a transcription result from raw segments.
    internal init(
        segments: [RawSegment],
        detectedLanguage: Language?,
        timings: RawTimings?
    ) {
        self.segments = segments.map { TranscriptionSegment(from: $0) }
        self.text = self.segments.map(\.text).joined()
        self.detectedLanguage = detectedLanguage
        self.timings = timings.map { TranscriptionTimings(from: $0) }
    }
}

/// A single segment of transcribed text with timing.
public struct TranscriptionSegment: Sendable, Identifiable {
    /// Unique identifier for this segment.
    public let id: Int
    
    /// The transcribed text for this segment.
    public let text: String
    
    /// Start time in seconds from the beginning of audio.
    public let startTime: TimeInterval
    
    /// End time in seconds from the beginning of audio.
    public let endTime: TimeInterval
    
    /// Individual tokens with their probabilities (if requested).
    public let tokens: [Token]?
    
    /// Probability that this segment contains no speech.
    public let noSpeechProbability: Float
    
    /// Whether this segment indicates a speaker turn.
    public let isSpeakerTurn: Bool
    
    /// Creates a segment from raw data.
    internal init(from raw: RawSegment) {
        self.id = raw.index
        self.text = raw.text
        self.startTime = raw.startTime
        self.endTime = raw.endTime
        self.noSpeechProbability = raw.noSpeechProbability
        self.isSpeakerTurn = raw.isSpeakerTurn
        self.tokens = raw.tokens?.map { Token(from: $0) }
    }
}

/// A single token from the transcription.
public struct Token: Sendable, Identifiable {
    /// Token ID in the vocabulary.
    public let id: Int
    
    /// The text content of the token.
    public let text: String
    
    /// Probability of this token (0.0 to 1.0).
    public let probability: Float
    
    /// Start time in seconds (if token timestamps enabled).
    public let startTime: TimeInterval?
    
    /// End time in seconds (if token timestamps enabled).
    public let endTime: TimeInterval?
    
    /// Creates a token from raw data.
    internal init(from raw: RawToken) {
        self.id = raw.id
        self.text = raw.text
        self.probability = raw.probability
        self.startTime = raw.startTime
        self.endTime = raw.endTime
    }
}

/// Performance timing information.
public struct TranscriptionTimings: Sendable {
    /// Time spent sampling in milliseconds.
    public let sampleMs: Float
    
    /// Time spent encoding in milliseconds.
    public let encodeMs: Float
    
    /// Time spent decoding in milliseconds.
    public let decodeMs: Float
    
    /// Total time in milliseconds.
    public let totalMs: Float
    
    /// Creates timings from raw data.
    internal init(from raw: RawTimings) {
        self.sampleMs = raw.sampleMs
        self.encodeMs = raw.encodeMs
        self.decodeMs = raw.decodeMs
        self.totalMs = raw.totalMs
    }
}
