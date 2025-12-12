import Foundation
import AVFoundation

/// A transcriber for real-time audio streaming with Voice Activity Detection.
///
/// This is the primary API for real-time microphone transcription. It uses VAD
/// to detect speech segments and emits transcription results as an AsyncSequence.
///
/// ```swift
/// let transcriber = try await StreamingTranscriber(
///     modelPath: modelURL,
///     vadModelPath: vadModelURL
/// )
///
/// // Start processing and iterate over segments
/// try await transcriber.start()
///
/// for try await segment in transcriber.segments {
///     print("[\(segment.startTime)]: \(segment.text)")
/// }
/// ```
///
/// Feed audio samples from AVAudioEngine:
/// ```swift
/// audioEngine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, time in
///     Task {
///         try await transcriber.process(buffer: buffer)
///     }
/// }
/// ```
public final class StreamingTranscriber: Sendable {
    
    // MARK: - Properties
    
    /// The underlying whisper context.
    private let whisperContext: WhisperContext
    
    /// The VAD context for speech detection (optional).
    private let vadContext: VADContext?
    
    /// The audio buffer for accumulating samples.
    private let audioBuffer: AudioRingBuffer
    
    /// Internal state management actor.
    private let stateManager: StreamingStateManager
    
    /// Transcription options.
    private let options: TranscriptionOptions
    
    /// VAD options (used when VAD model is provided).
    private let vadOptions: VADOptions
    
    /// Silence detector options (used when no VAD model is provided).
    private let silenceDetectorOptions: SilenceDetectorOptions
    
    /// Minimum audio duration (in seconds) before attempting transcription.
    private let minAudioDuration: TimeInterval
    
    /// Maximum audio duration (in seconds) before forcing transcription.
    private let maxAudioDuration: TimeInterval
    
    /// The stream continuation for emitting segments.
    private let segmentContinuation: AsyncThrowingStream<TranscriptionSegment, Error>.Continuation
    
    /// The public async stream of transcribed segments.
    public let segments: AsyncThrowingStream<TranscriptionSegment, Error>
    
    // MARK: - Initialization
    
    /// Creates a streaming transcriber with the specified model.
    ///
    /// - Parameters:
    ///   - modelPath: Path to the whisper.cpp GGML model file.
    ///   - vadModelPath: Optional path to the Silero VAD model file.
    ///     If provided, neural VAD will be used for speech detection.
    ///     If nil, lightweight RMS-based silence detection is used instead.
    ///   - configuration: Hardware and processing configuration.
    ///   - options: Transcription options.
    ///   - vadOptions: VAD options (only used if vadModelPath is provided).
    ///   - silenceDetectorOptions: Silence detection options (used when no VAD model).
    ///   - minAudioDuration: Minimum audio duration before transcription (default: 1.0s).
    ///   - maxAudioDuration: Maximum audio duration before forcing transcription (default: 30.0s).
    /// - Throws: `WhisperError.modelNotFound` or `WhisperError.modelLoadFailed`
    public init(
        modelPath: URL,
        vadModelPath: URL? = nil,
        configuration: WhisperConfiguration = .default,
        options: TranscriptionOptions = .default,
        vadOptions: VADOptions = .default,
        silenceDetectorOptions: SilenceDetectorOptions = .default,
        minAudioDuration: TimeInterval = 1.0,
        maxAudioDuration: TimeInterval = 30.0
    ) async throws {
        // Initialize whisper context
        self.whisperContext = try await Task {
            try WhisperContext(modelPath: modelPath, configuration: configuration)
        }.value
        
        // Initialize VAD context if model path provided
        if let vadPath = vadModelPath {
            self.vadContext = try await Task {
                try VADContext(
                    modelPath: vadPath,
                    useGPU: configuration.useGPU,
                    threadCount: Int(configuration.optimalThreadCount)
                )
            }.value
        } else {
            self.vadContext = nil
        }
        
        // Initialize other components
        self.audioBuffer = AudioRingBuffer(sampleRate: AudioProcessor.requiredSampleRate)
        self.stateManager = StreamingStateManager()
        self.options = options
        self.vadOptions = vadOptions
        self.silenceDetectorOptions = silenceDetectorOptions
        self.minAudioDuration = minAudioDuration
        self.maxAudioDuration = maxAudioDuration
        
        // Create the async stream for segments
        var continuation: AsyncThrowingStream<TranscriptionSegment, Error>.Continuation!
        self.segments = AsyncThrowingStream { cont in
            continuation = cont
        }
        self.segmentContinuation = continuation
    }
    
    // MARK: - Lifecycle
    
    /// Starts the streaming session.
    ///
    /// Must be called before processing audio. The transcriber will begin
    /// accepting audio samples and emitting segments.
    ///
    /// - Throws: `WhisperError.invalidState` if already started.
    public func start() async throws {
        let currentState = await stateManager.state
        guard currentState == .idle else {
            throw WhisperError.invalidState(
                expected: "idle",
                actual: currentState.description
            )
        }
        
        await stateManager.setState(.running)
        await audioBuffer.reset()
    }
    
    /// Stops the streaming session and finalizes transcription.
    ///
    /// Any remaining audio in the buffer will be processed before stopping.
    /// After calling this, `segments` will complete.
    ///
    /// - Returns: The complete transcription result from the final processing.
    /// - Throws: `WhisperError.invalidState` if not running.
    @discardableResult
    public func stop() async throws -> TranscriptionResult {
        let currentState = await stateManager.state
        guard currentState == .running else {
            throw WhisperError.invalidState(
                expected: "running",
                actual: currentState.description
            )
        }
        
        await stateManager.setState(.stopping)
        
        // Process any remaining audio
        let result = try await processFinalAudio()
        
        await stateManager.setState(.stopped)
        segmentContinuation.finish()
        
        return result
    }
    
    /// The current state of the transcriber.
    public var state: StreamingState {
        get async {
            await stateManager.state
        }
    }
    
    // MARK: - Audio Processing
    
    /// Processes an audio buffer from AVAudioEngine.
    ///
    /// The buffer will be automatically converted to the required format
    /// (16kHz mono Float32). Call this method from your AVAudioEngine tap.
    ///
    /// - Parameter buffer: Audio buffer from AVAudioEngine.
    /// - Throws: `WhisperError.invalidState` if `start()` hasn't been called.
    public func process(buffer: AVAudioPCMBuffer) async throws {
        let currentState = await stateManager.state
        guard currentState == .running else {
            throw WhisperError.invalidState(
                expected: "running",
                actual: currentState.description
            )
        }
        
        // Convert buffer to required format
        let samples = try AudioProcessor.convert(buffer, sampleRate: buffer.format.sampleRate)
        
        // Process the samples
        try await process(samples: samples)
    }
    
    /// Processes raw audio samples.
    ///
    /// - Parameters:
    ///   - samples: Audio samples as Float32 values (-1.0 to 1.0).
    ///   - sampleRate: Sample rate of the input audio. Defaults to 16000 Hz.
    /// - Throws: `WhisperError.invalidState` if `start()` hasn't been called.
    public func process(samples: [Float], sampleRate: Double = 16000) async throws {
        let currentState = await stateManager.state
        guard currentState == .running else {
            throw WhisperError.invalidState(
                expected: "running",
                actual: currentState.description
            )
        }
        
        // Resample if needed
        var processedSamples = samples
        if abs(sampleRate - AudioProcessor.requiredSampleRate) > 1.0 {
            processedSamples = AudioProcessor.resample(
                samples,
                from: sampleRate,
                to: AudioProcessor.requiredSampleRate
            )
        }
        
        // Append to buffer
        await audioBuffer.append(processedSamples)
        
        // Check if we have enough audio to process
        let duration = await audioBuffer.duration
        if duration >= minAudioDuration {
            try await attemptTranscription()
        }
    }
    
    // MARK: - Private Methods
    
    /// Attempts to transcribe the current buffer contents.
    private func attemptTranscription() async throws {
        let samples = await audioBuffer.currentSamples
        guard !samples.isEmpty else { return }
        
        // If VAD is available, use neural VAD for speech detection
        if let vad = vadContext {
            try await transcribeWithVAD(samples: samples, vad: vad)
        } else {
            // Use lightweight silence detection for chunking
            try await transcribeWithSilenceDetection(samples: samples)
        }
    }
    
    /// Transcribes audio using VAD to detect speech segments.
    private func transcribeWithVAD(samples: [Float], vad: VADContext) async throws {
        // Get speech segments from VAD
        let speechSegments = try await vad.getSpeechSegments(
            samples: samples,
            options: vadOptions
        )
        
        guard !speechSegments.isEmpty else {
            // No speech detected, keep accumulating
            return
        }
        
        // Process each complete speech segment
        for segment in speechSegments {
            // Calculate sample indices
            let startSample = Int(segment.startTime * Float(AudioProcessor.requiredSampleRate))
            let endSample = Int(segment.endTime * Float(AudioProcessor.requiredSampleRate))
            
            guard startSample < samples.count && startSample < endSample else { continue }
            
            let segmentSamples = Array(samples[startSample..<min(endSample, samples.count)])
            
            // Transcribe this segment
            let rawSegments = try await whisperContext.transcribe(
                samples: segmentSamples,
                options: options
            )
            
            // Emit each transcribed segment
            for rawSegment in rawSegments {
                let transcriptionSegment = TranscriptionSegment(from: rawSegment)
                segmentContinuation.yield(transcriptionSegment)
            }
        }
        
        // Consume the processed audio up to the last segment end
        if let lastSegment = speechSegments.last {
            let consumeCount = Int(lastSegment.endTime * Float(AudioProcessor.requiredSampleRate))
            _ = await audioBuffer.consume(consumeCount)
        }
    }
    
    /// Transcribes audio using lightweight silence detection for chunking.
    private func transcribeWithSilenceDetection(samples: [Float]) async throws {
        let duration = Double(samples.count) / AudioProcessor.requiredSampleRate
        
        // Check if audio contains speech
        guard SilenceDetector.containsSpeech(in: samples, threshold: silenceDetectorOptions.threshold) else {
            // No speech detected - if we have too much silence, clear the buffer
            if duration > maxAudioDuration {
                _ = await audioBuffer.consumeAll()
            }
            return
        }
        
        // If we haven't reached max duration, look for a silence break point
        if duration < maxAudioDuration {
            // Try to find a good break point in the audio
            if let breakPoint = SilenceDetector.findSilenceBreak(
                in: samples,
                options: silenceDetectorOptions
            ) {
                // We found a silence gap - transcribe up to that point
                let chunkSamples = Array(samples[..<breakPoint])
                try await transcribeChunk(chunkSamples)
                
                // Consume the processed audio
                _ = await audioBuffer.consume(breakPoint)
            }
            // If no break point found, keep accumulating
        } else {
            // Max duration reached - force transcription
            // Find best break point, or use entire buffer
            let breakPoint = SilenceDetector.findSilenceBreak(
                in: samples,
                options: silenceDetectorOptions
            ) ?? samples.count
            
            let chunkSamples = Array(samples[..<breakPoint])
            try await transcribeChunk(chunkSamples)
            
            // Consume processed audio, keeping overlap for context
            let overlapSamples = Int(0.5 * AudioProcessor.requiredSampleRate)
            let consumeCount = max(0, breakPoint - overlapSamples)
            _ = await audioBuffer.consume(consumeCount)
        }
    }
    
    /// Transcribes a chunk of audio and emits segments.
    private func transcribeChunk(_ samples: [Float]) async throws {
        guard !samples.isEmpty else { return }
        
        let rawSegments = try await whisperContext.transcribe(
            samples: samples,
            options: options
        )
        
        for rawSegment in rawSegments {
            let transcriptionSegment = TranscriptionSegment(from: rawSegment)
            segmentContinuation.yield(transcriptionSegment)
        }
    }
    
    /// Processes any remaining audio when stopping.
    private func processFinalAudio() async throws -> TranscriptionResult {
        var samples = await audioBuffer.consumeAll()
        
        guard !samples.isEmpty else {
            return TranscriptionResult(
                segments: [],
                detectedLanguage: nil,
                timings: nil
            )
        }
        
        // Pad to minimum length if needed (whisper.cpp requires >= 100ms)
        samples = AudioProcessor.padToMinimumLength(samples)
        
        // Transcribe remaining audio
        let rawSegments = try await whisperContext.transcribe(
            samples: samples,
            options: options
        )
        
        // Emit final segments
        for rawSegment in rawSegments {
            let transcriptionSegment = TranscriptionSegment(from: rawSegment)
            segmentContinuation.yield(transcriptionSegment)
        }
        
        // Get detected language
        let detectedLanguage = await getDetectedLanguage()
        let timings = await whisperContext.getTimings()
        
        return TranscriptionResult(
            segments: rawSegments,
            detectedLanguage: detectedLanguage,
            timings: timings
        )
    }
    
    /// Gets the detected language from the context if auto-detection was used.
    private func getDetectedLanguage() async -> Language? {
        guard options.language == nil || options.language == .auto else {
            return nil
        }
        
        let langId = await whisperContext.detectedLanguageId
        guard langId >= 0,
              let langString = await whisperContext.languageString(for: langId) else {
            return nil
        }
        
        return Language(rawValue: langString)
    }
}

// MARK: - State Manager

/// Internal actor for managing streaming state.
private actor StreamingStateManager {
    var state: StreamingState = .idle
    
    func setState(_ newState: StreamingState) {
        state = newState
    }
}
