import Foundation
import AVFoundation

/// A streaming transcriber that automatically handles audio input from an AVAudioInputNode.
///
/// This is a higher-level API compared to `StreamingTranscriber` - you provide the input node
/// and the transcriber handles installing taps, audio conversion, and buffer management.
///
/// **Example Usage:**
/// ```swift
/// let engine = AVAudioEngine()
/// let input = engine.inputNode
///
/// let transcriber = try await BetaStreamingTranscriber(
///     modelPath: modelURL,
///     input: input
/// )
///
/// // Start the audio engine
/// try engine.start()
///
/// // Start transcription
/// try await transcriber.start()
///
/// // Consume segments as they arrive
/// for try await segment in transcriber.segments {
///     print(segment.text)
/// }
///
/// // Or stop manually and get the final result
/// let finalResult = try await transcriber.stop()
/// print(finalResult.text)
/// ```
///
/// - Note: You are responsible for managing the `AVAudioEngine` lifecycle (preparing, starting,
///   stopping). The transcriber only manages its tap on the input node.
public final class BetaStreamingTranscriber: @unchecked Sendable {
    
    // MARK: - Configuration
    
    /// Configuration options for the streaming transcriber.
    public struct Configuration: Sendable {
        /// Transcription options (language, sampling strategy, etc.).
        public var transcriptionOptions: TranscriptionOptions
        
        /// Hardware configuration (GPU, threads, etc.).
        public var whisperConfiguration: WhisperConfiguration
        
        /// VAD options (only used if VAD model is provided).
        public var vadOptions: VADOptions
        
        /// Silence detector options (used when no VAD model is provided).
        public var silenceDetectorOptions: SilenceDetectorOptions
        
        /// Minimum audio duration before attempting transcription.
        public var minAudioDuration: TimeInterval
        
        /// Maximum audio duration before forcing transcription.
        public var maxAudioDuration: TimeInterval
        
        /// Duration of each audio buffer callback (in seconds).
        /// This is converted to frames based on the hardware sample rate.
        /// Smaller values = lower latency but more CPU overhead.
        /// Larger values = higher latency but more efficient.
        public var bufferDuration: TimeInterval
        
        /// Default configuration.
        public static let `default` = Configuration(
            transcriptionOptions: .default,
            whisperConfiguration: .default,
            vadOptions: .default,
            silenceDetectorOptions: .default,
            minAudioDuration: 1.0,
            maxAudioDuration: 30.0,
            bufferDuration: 0.1 // 100ms
        )
        
        public init(
            transcriptionOptions: TranscriptionOptions = .default,
            whisperConfiguration: WhisperConfiguration = .default,
            vadOptions: VADOptions = .default,
            silenceDetectorOptions: SilenceDetectorOptions = .default,
            minAudioDuration: TimeInterval = 1.0,
            maxAudioDuration: TimeInterval = 30.0,
            bufferDuration: TimeInterval = 0.1
        ) {
            self.transcriptionOptions = transcriptionOptions
            self.whisperConfiguration = whisperConfiguration
            self.vadOptions = vadOptions
            self.silenceDetectorOptions = silenceDetectorOptions
            self.minAudioDuration = minAudioDuration
            self.maxAudioDuration = maxAudioDuration
            self.bufferDuration = bufferDuration
        }
    }
    
    // MARK: - Properties
    
    /// The audio input node to read from.
    private let inputNode: AVAudioInputNode
    
    /// The whisper context for transcription.
    private let whisperContext: WhisperContext
    
    /// The VAD context for speech detection (optional).
    private let vadContext: VADContext?
    
    /// The audio buffer for accumulating samples.
    private let audioBuffer: AudioRingBuffer
    
    /// Internal state manager.
    private let stateManager: StreamingStateManager
    
    /// Configuration for this transcriber.
    private let configuration: Configuration
    
    /// The stream continuation for emitting segments.
    private let segmentContinuation: AsyncThrowingStream<TranscriptionSegment, Error>.Continuation
    
    /// The public async stream of transcribed segments.
    public let segments: AsyncThrowingStream<TranscriptionSegment, Error>
    
    /// Processing task handle.
    private let processingTaskHolder: ProcessingTaskHolder
    
    // MARK: - Initialization
    
    /// Creates a new streaming transcriber.
    ///
    /// - Parameters:
    ///   - modelPath: Path to the whisper.cpp GGML model file.
    ///   - vadModelPath: Optional path to the Silero VAD model file.
    ///     If provided, neural VAD will be used for speech detection.
    ///     If nil, lightweight RMS-based silence detection is used instead.
    ///   - input: The audio input node to read from.
    ///   - configuration: Configuration options for the transcriber.
    /// - Throws: `WhisperError.modelNotFound` or `WhisperError.modelLoadFailed`
    public init(
        modelPath: URL,
        vadModelPath: URL? = nil,
        input: AVAudioInputNode,
        configuration: Configuration = .default
    ) async throws {
        self.inputNode = input
        self.configuration = configuration
        
        // Initialize whisper context
        self.whisperContext = try await Task {
            try WhisperContext(
                modelPath: modelPath,
                configuration: configuration.whisperConfiguration
            )
        }.value
        
        // Initialize VAD context if model path provided
        if let vadPath = vadModelPath {
            self.vadContext = try await Task {
                try VADContext(
                    modelPath: vadPath,
                    useGPU: configuration.whisperConfiguration.useGPU,
                    threadCount: Int(configuration.whisperConfiguration.optimalThreadCount)
                )
            }.value
        } else {
            self.vadContext = nil
        }
        
        // Initialize other components
        self.audioBuffer = AudioRingBuffer(sampleRate: AudioProcessor.requiredSampleRate)
        self.stateManager = StreamingStateManager()
        self.processingTaskHolder = ProcessingTaskHolder()
        
        // Create the async stream for segments
        var continuation: AsyncThrowingStream<TranscriptionSegment, Error>.Continuation!
        self.segments = AsyncThrowingStream { cont in
            continuation = cont
        }
        self.segmentContinuation = continuation
    }
    
    // MARK: - Lifecycle
    
    /// Starts the streaming transcription session.
    ///
    /// This installs a tap on the audio input node and begins processing audio.
    /// The audio engine must already be running when you call this method.
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
        await stateManager.clearHistory()
        await audioBuffer.reset()
        
        // Get the input format and calculate buffer size based on hardware sample rate
        let format = inputNode.outputFormat(forBus: 0)
        let bufferSize = AVAudioFrameCount(format.sampleRate * configuration.bufferDuration)
        
        // Install the tap
        inputNode.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: format
        ) { [weak self] buffer, _ in
            guard let self = self else { return }
            
            // Convert samples synchronously before crossing async boundary
            // AVAudioPCMBuffer is not Sendable, so we must extract the data here
            do {
                let samples = try AudioProcessor.convert(buffer, sampleRate: format.sampleRate)
                Task {
                    await self.appendSamples(samples)
                }
            } catch {
                // Audio conversion error - skip this buffer
            }
        }
        
        // Start the processing loop
        let task = Task { [weak self] in
            guard let self = self else { return }
            await self.processingLoop()
        }
        await processingTaskHolder.setTask(task)
    }
    
    /// Stops the streaming transcription session.
    ///
    /// Removes the tap from the input node and processes any remaining audio.
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
        
        // Remove the tap
        inputNode.removeTap(onBus: 0)
        
        // Cancel processing task
        await processingTaskHolder.cancel()
        
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
    
    // MARK: - Private Methods
    
    /// Appends samples to the audio buffer if currently running.
    private func appendSamples(_ samples: [Float]) async {
        let currentState = await stateManager.state
        guard currentState == .running else { return }
        await audioBuffer.append(samples)
    }
    
    /// Main processing loop that checks for transcription opportunities.
    private func processingLoop() async {
        while await stateManager.state == .running {
            // Check buffer duration
            let duration = await audioBuffer.duration
            
            if duration >= configuration.minAudioDuration {
                do {
                    try await attemptTranscription()
                } catch {
                    // Set failed state and emit error
                    await stateManager.setState(.failed(error as? WhisperError ?? .transcriptionFailed(error.localizedDescription)))
                    segmentContinuation.finish(throwing: error)
                    return
                }
            }
            
            // Small delay to avoid busy waiting
            try? await Task.sleep(for: .milliseconds(100)) // 100ms
        }
    }
    
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
            options: configuration.vadOptions
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
                options: configuration.transcriptionOptions
            )
            
            // Emit each transcribed segment
            for rawSegment in rawSegments {
                let transcriptionSegment = TranscriptionSegment(from: rawSegment)
                // Only emit if not a duplicate
                if await stateManager.shouldEmit(text: transcriptionSegment.text) {
                    segmentContinuation.yield(transcriptionSegment)
                }
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
        guard SilenceDetector.containsSpeech(
            in: samples,
            threshold: configuration.silenceDetectorOptions.threshold
        ) else {
            // No speech detected - if we have too much silence, clear the buffer
            if duration > configuration.maxAudioDuration {
                _ = await audioBuffer.consumeAll()
            }
            return
        }
        
        // If we haven't reached max duration, look for a silence break point
        if duration < configuration.maxAudioDuration {
            // Try to find a good break point in the audio
            if let breakPoint = SilenceDetector.findSilenceBreak(
                in: samples,
                options: configuration.silenceDetectorOptions
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
                options: configuration.silenceDetectorOptions
            ) ?? samples.count
            
            let chunkSamples = Array(samples[..<breakPoint])
            try await transcribeChunk(chunkSamples)
            
            // Consume processed audio, keeping minimal overlap
            let overlapSamples = Int(0.1 * AudioProcessor.requiredSampleRate)
            let consumeCount = max(0, breakPoint - overlapSamples)
            _ = await audioBuffer.consume(consumeCount)
        }
    }
    
    /// Transcribes a chunk of audio and emits segments.
    private func transcribeChunk(_ samples: [Float]) async throws {
        guard !samples.isEmpty else { return }
        
        let rawSegments = try await whisperContext.transcribe(
            samples: samples,
            options: configuration.transcriptionOptions
        )
        
        for rawSegment in rawSegments {
            let transcriptionSegment = TranscriptionSegment(from: rawSegment)
            // Only emit if not a duplicate
            if await stateManager.shouldEmit(text: transcriptionSegment.text) {
                segmentContinuation.yield(transcriptionSegment)
            }
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
            options: configuration.transcriptionOptions
        )
        
        // Emit final segments
        for rawSegment in rawSegments {
            let transcriptionSegment = TranscriptionSegment(from: rawSegment)
            // Only emit if not a duplicate
            if await stateManager.shouldEmit(text: transcriptionSegment.text) {
                segmentContinuation.yield(transcriptionSegment)
            }
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
        let options = configuration.transcriptionOptions
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

// MARK: - Processing Task Holder

/// Actor to hold the processing task reference.
private actor ProcessingTaskHolder {
    private var task: Task<Void, Never>?
    
    func setTask(_ task: Task<Void, Never>) {
        self.task = task
    }
    
    func cancel() {
        task?.cancel()
        task = nil
    }
}
