import Foundation

/// A transcriber for converting audio files to text.
///
/// Use this for batch processing of audio files. For real-time microphone
/// transcription, use ``StreamingTranscriber`` instead.
///
/// ```swift
/// let transcriber = try await Transcriber(modelPath: modelURL)
/// let result = try await transcriber.transcribe(file: audioURL)
/// print(result.text)
/// ```
public final class Transcriber: Sendable {
    
    /// The underlying whisper context.
    private let context: WhisperContext
    
    /// Creates a transcriber with the specified model.
    /// - Parameters:
    ///   - modelPath: Path to the whisper.cpp GGML model file.
    ///   - configuration: Hardware and processing configuration.
    /// - Throws: `WhisperError.modelNotFound` if the model file doesn't exist,
    ///           or `WhisperError.modelLoadFailed` if the model cannot be loaded.
    public init(
        modelPath: URL,
        configuration: WhisperConfiguration = .default
    ) async throws {
        self.context = try await Task {
            try WhisperContext(modelPath: modelPath, configuration: configuration)
        }.value
    }
    
    // MARK: - Transcription Methods
    
    /// Transcribes an audio file.
    /// - Parameters:
    ///   - file: URL to the audio file (WAV, MP3, M4A, CAF, etc.).
    ///   - options: Transcription options.
    /// - Returns: The transcription result with text and segments.
    /// - Throws: `WhisperError.audioLoadFailed` if the file cannot be read,
    ///           or `WhisperError.transcriptionFailed` on transcription failure.
    public func transcribe(
        file: URL,
        options: TranscriptionOptions = .default
    ) async throws -> TranscriptionResult {
        // Load and convert audio to required format
        let samples = try AudioProcessor.loadAudioFile(file)
        
        // Transcribe the samples
        return try await transcribe(samples: samples, options: options)
    }
    
    /// Transcribes raw audio samples.
    /// - Parameters:
    ///   - samples: Audio samples as Float32 values (-1.0 to 1.0), must be 16kHz mono.
    ///   - options: Transcription options.
    /// - Returns: The transcription result with text and segments.
    /// - Throws: `WhisperError.transcriptionFailed` on failure.
    public func transcribe(
        samples: [Float],
        options: TranscriptionOptions = .default
    ) async throws -> TranscriptionResult {
        guard !samples.isEmpty else {
            return TranscriptionResult(
                segments: [],
                detectedLanguage: nil,
                timings: nil
            )
        }
        
        // Run transcription on the actor
        let rawSegments = try await context.transcribe(samples: samples, options: options)
        
        // Get detected language if auto-detection was used
        let detectedLanguage = await getDetectedLanguage(options: options)
        
        // Get timing information
        let timings = await context.getTimings()
        
        return TranscriptionResult(
            segments: rawSegments,
            detectedLanguage: detectedLanguage,
            timings: timings
        )
    }
    
    // MARK: - Private Helpers
    
    /// Gets the detected language from the context if auto-detection was used.
    private func getDetectedLanguage(options: TranscriptionOptions) async -> Language? {
        // Only return detected language if auto-detection was used
        guard options.language == nil || options.language == .auto else {
            return nil
        }
        
        let langId = await context.detectedLanguageId
        guard langId >= 0,
              let langString = await context.languageString(for: langId) else {
            return nil
        }
        
        return Language(rawValue: langString)
    }
}
