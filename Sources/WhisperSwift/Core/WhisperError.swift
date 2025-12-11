import Foundation

/// Errors that can occur during whisper.cpp operations.
public enum WhisperError: Error, Sendable {
    /// The model file was not found at the specified path.
    case modelNotFound(URL)
    
    /// The model failed to load (invalid format, corrupted, etc.).
    case modelLoadFailed(String)
    
    /// The audio file could not be read or decoded.
    case audioLoadFailed(URL, underlying: Error)
    
    /// The audio format is not supported or couldn't be converted.
    case invalidAudioFormat(String)
    
    /// Transcription failed during processing.
    case transcriptionFailed(String)
    
    /// The streaming transcriber is not in the correct state.
    case invalidState(expected: String, actual: String)
    
    /// VAD model failed to load or process.
    case vadFailed(String)
    
    /// An internal whisper.cpp error occurred.
    case internalError(String)
}

extension WhisperError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let url):
            return "Model file not found at: \(url.path)"
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .audioLoadFailed(let url, let underlying):
            return "Failed to load audio file at \(url.path): \(underlying.localizedDescription)"
        case .invalidAudioFormat(let reason):
            return "Invalid audio format: \(reason)"
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        case .invalidState(let expected, let actual):
            return "Invalid transcriber state: expected \(expected), got \(actual)"
        case .vadFailed(let reason):
            return "VAD processing failed: \(reason)"
        case .internalError(let reason):
            return "Internal whisper.cpp error: \(reason)"
        }
    }
}
