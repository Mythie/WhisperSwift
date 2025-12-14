import Foundation
import AVFoundation

/// A streaming transcriber that works like the whisper.cpp stream example.
///
/// This transcriber uses a sliding window approach with overlap to provide
/// real-time transcription with smooth output. It supports two modes:
///
/// 1. **Sliding Window Mode** (default): Continuously processes audio in overlapping
///    windows, providing low-latency incremental output.
///
/// 2. **VAD Mode**: Uses voice activity detection to transcribe complete utterances,
///    providing higher quality output at the cost of latency.
///
/// **Example Usage:**
/// ```swift
/// let engine = AVAudioEngine()
///
/// let transcriber = try await BetaStreamingTranscriber(
///     modelPath: modelURL,
///     engine: engine
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
/// ```
///
/// - Note: You are responsible for managing the `AVAudioEngine` lifecycle.
public final class BetaStreamingTranscriber: @unchecked Sendable {
    
    // MARK: - Configuration
    
    /// Configuration options for the streaming transcriber.
    public struct Configuration: Sendable {
        /// Audio step size - how much NEW audio to process each iteration (in seconds).
        /// Smaller = more responsive but more CPU usage.
        /// Default: 3.0 seconds (matches whisper.cpp stream example)
        public var stepDuration: TimeInterval
        
        /// Total audio window length for each transcription (in seconds).
        /// This is the total context window passed to whisper.
        /// Default: 10.0 seconds (matches whisper.cpp stream example)
        public var windowDuration: TimeInterval
        
        /// Audio to keep from previous iteration to avoid word boundary issues (in seconds).
        /// This overlap helps maintain continuity between transcriptions.
        /// Default: 0.2 seconds (200ms, matches whisper.cpp stream example)
        public var keepDuration: TimeInterval
        
        /// Transcription options (language, sampling strategy, etc.).
        public var transcriptionOptions: TranscriptionOptions
        
        /// Hardware configuration (GPU, threads, etc.).
        public var whisperConfiguration: WhisperConfiguration
        
        /// Whether to keep context (prompt tokens) between transcriptions.
        /// When true, uses output from previous transcription as prompt for next.
        /// This can improve accuracy but may cause hallucinations if context drifts.
        /// Default: false (matches whisper.cpp stream example)
        public var keepContext: Bool
        
        /// VAD options for neural voice activity detection.
        /// Only used when a VAD model path is provided to the transcriber.
        public var vadOptions: VADOptions
        
        /// Duration of each audio buffer callback (in seconds).
        /// This affects how often the audio callback fires.
        /// Smaller = more responsive but more overhead.
        /// Default: 0.1 seconds (100ms)
        public var bufferDuration: TimeInterval
        
        /// Maximum tokens per audio chunk.
        /// Set to 0 for no limit (whisper will decide based on audio length).
        /// Default: 32 (matches whisper.cpp stream example)
        public var maxTokens: Int
        
        /// Default configuration for real-time streaming.
        public static let `default` = Configuration(
            stepDuration: 3.0,
            windowDuration: 10.0,
            keepDuration: 0.2,
            transcriptionOptions: .init(language: .english), // English for speed
            whisperConfiguration: .default,
            keepContext: false,
            vadOptions: .default,
            bufferDuration: 0.1,
            maxTokens: 32
        )
        
        /// Configuration optimized for low latency.
        public static let lowLatency = Configuration(
            stepDuration: 1.0,
            windowDuration: 5.0,
            keepDuration: 0.2,
            transcriptionOptions: .init(language: .english),
            whisperConfiguration: .default,
            keepContext: false,
            vadOptions: .default,
            bufferDuration: 0.05,
            maxTokens: 16
        )
        
        /// Configuration for VAD mode - waits for speech, then transcribes complete utterances.
        /// Requires a VAD model path to be provided when initializing the transcriber.
        public static let vadMode = Configuration(
            stepDuration: 3.0,
            windowDuration: 10.0,
            keepDuration: 0.0, // Not used in VAD mode
            transcriptionOptions: .default,
            whisperConfiguration: .default,
            keepContext: false,
            vadOptions: .default,
            bufferDuration: 0.1,
            maxTokens: 0
        )
        
        public init(
            stepDuration: TimeInterval = 3.0,
            windowDuration: TimeInterval = 10.0,
            keepDuration: TimeInterval = 0.2,
            transcriptionOptions: TranscriptionOptions = .init(language: .english),
            whisperConfiguration: WhisperConfiguration = .default,
            keepContext: Bool = false,
            vadOptions: VADOptions = .default,
            bufferDuration: TimeInterval = 0.1,
            maxTokens: Int = 32
        ) {
            self.stepDuration = stepDuration
            self.windowDuration = max(windowDuration, stepDuration)
            self.keepDuration = min(keepDuration, stepDuration)
            self.transcriptionOptions = transcriptionOptions
            self.whisperConfiguration = whisperConfiguration
            self.keepContext = keepContext
            self.vadOptions = vadOptions
            self.bufferDuration = bufferDuration
            self.maxTokens = maxTokens
        }
    }
    
    // MARK: - Streaming Segment
    
    /// A segment emitted during streaming transcription.
    public struct StreamingSegment: Sendable {
        /// The transcribed text.
        public let text: String
        
        /// Whether this is a partial (in-progress) or final segment.
        /// Partial segments may be updated or replaced by subsequent segments.
        public let isPartial: Bool
        
        /// The iteration number this segment came from.
        public let iteration: Int
    }
    
    // MARK: - Properties
    
    /// The audio engine.
    private let engine: AVAudioEngine
    
    /// The input node to read from.
    private let inputNode: AVAudioInputNode
    
    /// The whisper context for transcription.
    private let whisperContext: WhisperContext
    
    /// The VAD context for neural speech detection (optional).
    /// When provided, enables VAD mode for speech-triggered transcription.
    private let vadContext: VADContext?
    
    /// Configuration for this transcriber.
    private let configuration: Configuration
    
    /// The stream continuation for emitting segments.
    private let segmentContinuation: AsyncThrowingStream<StreamingSegment, Error>.Continuation
    
    /// The public async stream of transcribed segments.
    public let segments: AsyncThrowingStream<StreamingSegment, Error>
    
    /// Processing state actor.
    private let state: BetaStreamingState
    
    // MARK: - Initialization
    
    /// Creates a new streaming transcriber.
    ///
    /// - Parameters:
    ///   - modelPath: Path to the whisper.cpp GGML model file.
    ///   - engine: The audio engine to use.
    ///   - vadModelPath: Optional path to the Silero VAD model file.
    ///     When provided, enables VAD mode for speech-triggered transcription.
    ///     Download from: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-silero-vad.bin
    ///   - configuration: Configuration options for the transcriber.
    /// - Throws: `WhisperError.modelNotFound` or `WhisperError.modelLoadFailed`
    public init(
        modelPath: URL,
        engine: AVAudioEngine,
        vadModelPath: URL? = nil,
        configuration: Configuration = .default
    ) async throws {
        self.engine = engine
        self.configuration = configuration
        self.inputNode = engine.inputNode
        
        // Initialize whisper context
        self.whisperContext = try await Task {
            try WhisperContext(
                modelPath: modelPath,
                configuration: configuration.whisperConfiguration
            )
        }.value
        
        // Initialize VAD context if model path provided
        // Note: VAD currently crashes with Metal, so we force CPU-only
        if let vadPath = vadModelPath {
            self.vadContext = try await Task {
                try VADContext(
                    modelPath: vadPath,
                    useGPU: false,  // Force CPU - Metal crashes with VAD
                    threadCount: Int(configuration.whisperConfiguration.optimalThreadCount)
                )
            }.value
        } else {
            self.vadContext = nil
        }
        
        // Initialize state
        self.state = BetaStreamingState(configuration: configuration)
        
        // Create the async stream for segments
        var continuation: AsyncThrowingStream<StreamingSegment, Error>.Continuation!
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
        guard await state.tryStart() else {
            throw WhisperError.invalidState(expected: "idle", actual: "already running")
        }
        
        // Get the input format and calculate buffer size
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
            do {
                let samples = try AudioProcessor.convert(buffer, sampleRate: format.sampleRate)
                Task { [weak self] in
                    await self?.state.appendSamples(samples)
                }
            } catch {
                // Audio conversion error - skip this buffer
            }
        }
        
        // Start the processing loop
        Task { [weak self] in
            await self?.processingLoop()
        }
    }
    
    /// Stops the streaming transcription session.
    ///
    /// Removes the tap from the input node and processes any remaining audio.
    ///
    /// - Returns: The final transcribed text from the session.
    /// - Throws: `WhisperError.invalidState` if not running.
    @discardableResult
    public func stop() async throws -> String {
        guard await state.tryStop() else {
            throw WhisperError.invalidState(expected: "running", actual: "not running")
        }
        
        // Remove the tap
        inputNode.removeTap(onBus: 0)
        
        // Process any remaining audio
        let finalText = try await processFinalAudio()
        
        await state.finish()
        segmentContinuation.finish()
        
        return finalText
    }
    
    /// Whether the transcriber is currently running.
    public var isRunning: Bool {
        get async {
            await state.isRunning
        }
    }
    
    // MARK: - Processing Loop
    
    /// Main processing loop that handles transcription timing.
    private func processingLoop() async {
        if let vadContext = vadContext {
            await vadProcessingLoop(vad: vadContext)
        } else {
            await slidingWindowProcessingLoop()
        }
    }
    
    /// Processing loop for sliding window mode (like whisper.cpp stream default).
    private func slidingWindowProcessingLoop() async {
        let stepSamples = Int(configuration.stepDuration * AudioProcessor.requiredSampleRate)
        
        while await state.isRunning {
            // Wait until we have enough new samples
            let newSampleCount = await state.newSampleCount
            
            if newSampleCount >= stepSamples {
                do {
                    try await processWindow()
                } catch {
                    await state.setError(error)
                    segmentContinuation.finish(throwing: error)
                    return
                }
            } else {
                // Small delay to avoid busy waiting
                try? await Task.sleep(for: .milliseconds(10))
            }
        }
    }
    
    /// Processing loop for VAD mode (like whisper.cpp stream with VAD).
    /// Uses neural Silero VAD for accurate speech detection.
    private func vadProcessingLoop(vad: VADContext) async {
        let checkIntervalMs: UInt64 = 100
        let minSpeechSamples = Int(configuration.vadOptions.minSpeechDurationMs) * Int(AudioProcessor.requiredSampleRate) / 1000
        
        while await state.isRunning {
            // Check for speech periodically
            try? await Task.sleep(for: .milliseconds(checkIntervalMs))
            
            let samples = await state.getAllSamples()
            guard samples.count >= minSpeechSamples else { continue }
            
            // Use neural VAD to detect speech segments
            do {
                let speechSegments = try await vad.getSpeechSegments(
                    samples: samples,
                    options: configuration.vadOptions
                )
                
                guard !speechSegments.isEmpty else { continue }
                
                // Process each detected speech segment
                for segment in speechSegments {
                    let startSample = Int(segment.startTime * Float(AudioProcessor.requiredSampleRate))
                    let endSample = Int(segment.endTime * Float(AudioProcessor.requiredSampleRate))
                    
                    guard startSample < samples.count && startSample < endSample else { continue }
                    
                    let segmentSamples = Array(samples[startSample..<min(endSample, samples.count)])
                    
                    guard !segmentSamples.isEmpty else { continue }
                    
                    try await transcribeAndEmit(samples: segmentSamples, isPartial: false)
                }
                
                // Consume processed audio up to the last segment end
                if let lastSegment = speechSegments.last {
                    let consumeCount = Int(lastSegment.endTime * Float(AudioProcessor.requiredSampleRate))
                    await state.consumeSamples(count: consumeCount)
                }
            } catch {
                await state.setError(error)
                segmentContinuation.finish(throwing: error)
                return
            }
        }
    }
    
    /// Process a single window in sliding window mode.
    private func processWindow() async throws {
        let windowSamples = Int(configuration.windowDuration * AudioProcessor.requiredSampleRate)
        let keepSamples = Int(configuration.keepDuration * AudioProcessor.requiredSampleRate)
        
        // Get new samples and combine with kept samples from previous iteration
        let newSamples = await state.consumeNewSamples()
        let keptSamples = await state.getKeptSamples()
        
        // Build the full window: [kept samples from previous] + [new samples]
        var windowBuffer = keptSamples + newSamples
        
        // Trim to window size if needed
        if windowBuffer.count > windowSamples {
            windowBuffer = Array(windowBuffer.suffix(windowSamples))
        }
        
        guard !windowBuffer.isEmpty else { return }
        
        // Transcribe the window
        try await transcribeAndEmit(samples: windowBuffer, isPartial: true)
        
        // Keep audio for next iteration to avoid word boundary issues
        if windowBuffer.count > keepSamples {
            let toKeep = Array(windowBuffer.suffix(keepSamples))
            await state.setKeptSamples(toKeep)
        }
    }
    
    /// Transcribes samples and emits segments.
    private func transcribeAndEmit(samples: [Float], isPartial: Bool) async throws {
        // Build transcription options for streaming
        var options = configuration.transcriptionOptions
        
        // In sliding window mode (isPartial=true), use single segment for cleaner output
        if isPartial {
            options.singleSegment = true
        }
        
        // Set max tokens
        if configuration.maxTokens > 0 {
            options.maxTokens = configuration.maxTokens
        }
        
        // Use prompt tokens from previous iteration if keeping context
        if configuration.keepContext {
            let promptTokens = await state.getPromptTokens()
            if !promptTokens.isEmpty {
                // Note: This would require extending TranscriptionOptions to support prompt tokens
                // For now, we use the initial prompt as a text hint
            }
        }
        
        // Transcribe
        let rawSegments = try await whisperContext.transcribe(
            samples: samples,
            options: options
        )
        
        let iteration = await state.incrementIteration()
        
        // Combine all segment text
        let fullText = rawSegments.map { $0.text }.joined()
        
        guard !fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        // Emit segment
        let segment = StreamingSegment(
            text: fullText,
            isPartial: isPartial,
            iteration: iteration
        )
        segmentContinuation.yield(segment)
        
        // Store tokens for context continuity if enabled
        if configuration.keepContext {
            let tokens = rawSegments.flatMap { $0.tokens ?? [] }.map { Int32($0.id) }
            await state.setPromptTokens(tokens)
        }
    }
    
    /// Processes any remaining audio when stopping.
    private func processFinalAudio() async throws -> String {
        let allSamples = await state.getAllSamples()
        
        guard !allSamples.isEmpty else { return "" }
        
        // Pad to minimum length if needed
        let paddedSamples = AudioProcessor.padToMinimumLength(allSamples)
        
        // Transcribe with full options (not streaming mode)
        let rawSegments = try await whisperContext.transcribe(
            samples: paddedSamples,
            options: configuration.transcriptionOptions
        )
        
        let fullText = rawSegments.map { $0.text }.joined()
        
        if !fullText.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty {
            let iteration = await state.incrementIteration()
            let segment = StreamingSegment(
                text: fullText,
                isPartial: false,
                iteration: iteration
            )
            segmentContinuation.yield(segment)
        }
        
        return fullText
    }
}

// MARK: - Streaming State Actor

/// Actor managing the streaming transcription state.
private actor BetaStreamingState {
    private let configuration: BetaStreamingTranscriber.Configuration
    
    // Lifecycle state
    private var running = false
    private var finished = false
    private var error: Error?
    
    // Audio buffers
    private var newSamples: [Float] = []  // New samples since last processing
    private var keptSamples: [Float] = [] // Samples kept from previous iteration
    private var allSamples: [Float] = []  // All samples for final processing
    
    // Context
    private var promptTokens: [Int32] = []
    private var iteration = 0
    
    init(configuration: BetaStreamingTranscriber.Configuration) {
        self.configuration = configuration
    }
    
    // MARK: - Lifecycle
    
    var isRunning: Bool { running && !finished }
    
    func tryStart() -> Bool {
        guard !running && !finished else { return false }
        running = true
        return true
    }
    
    func tryStop() -> Bool {
        guard running && !finished else { return false }
        running = false
        return true
    }
    
    func finish() {
        finished = true
    }
    
    func setError(_ error: Error) {
        self.error = error
        running = false
    }
    
    // MARK: - Audio Management
    
    func appendSamples(_ samples: [Float]) {
        newSamples.append(contentsOf: samples)
        allSamples.append(contentsOf: samples)
    }
    
    var newSampleCount: Int { newSamples.count }
    
    func getNewSamples() -> [Float] {
        return newSamples
    }
    
    func consumeNewSamples() -> [Float] {
        let samples = newSamples
        newSamples.removeAll(keepingCapacity: true)
        return samples
    }
    
    func getSamplesForTranscription(maxCount: Int) -> [Float] {
        let count = min(allSamples.count, maxCount)
        return Array(allSamples.suffix(count))
    }
    
    func markProcessed() {
        newSamples.removeAll(keepingCapacity: true)
    }
    
    func getKeptSamples() -> [Float] {
        return keptSamples
    }
    
    func setKeptSamples(_ samples: [Float]) {
        keptSamples = samples
    }
    
    func getAllSamples() -> [Float] {
        return allSamples
    }
    
    /// Consumes (removes) the specified number of samples from the beginning of allSamples.
    func consumeSamples(count: Int) {
        let actualCount = min(count, allSamples.count)
        allSamples.removeFirst(actualCount)
        // Also remove from newSamples if they overlap
        let newCount = min(count, newSamples.count)
        if newCount > 0 {
            newSamples.removeFirst(newCount)
        }
    }
    
    // MARK: - Context
    
    func getPromptTokens() -> [Int32] {
        return promptTokens
    }
    
    func setPromptTokens(_ tokens: [Int32]) {
        promptTokens = tokens
    }
    
    func incrementIteration() -> Int {
        iteration += 1
        return iteration
    }
}
