// WhisperSwift - Idiomatic Swift bindings for whisper.cpp
//
// A high-quality Swift library for speech-to-text transcription using
// OpenAI's Whisper model via whisper.cpp.

@_exported import Foundation

// MARK: - Core
// Internal: WhisperContext (actor wrapping whisper_context)
// Internal: VADContext (actor wrapping whisper_vad_context)

// MARK: - Configuration
// Public: WhisperConfiguration, TranscriptionOptions, Language, SamplingStrategy
// Public: VADOptions

// MARK: - Transcription
// Public: Transcriber - Batch transcription API
// Public: StreamingTranscriber - Real-time streaming API
// Public: TranscriptionResult, TranscriptionSegment, Token, TranscriptionTimings
// Public: StreamingState

// MARK: - Audio
// Public: AudioProcessor - Audio format conversion utilities
// Internal: AudioRingBuffer - Streaming audio buffer

// MARK: - Errors
// Public: WhisperError

// MARK: - Constants

/// The required sample rate for whisper.cpp audio input.
public let whisperSampleRate: Double = 16000

/// The audio chunk size in seconds used by whisper.cpp.
public let whisperChunkSize: Int = 30
