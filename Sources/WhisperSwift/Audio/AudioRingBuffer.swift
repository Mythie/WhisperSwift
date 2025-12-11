import Foundation

/// A thread-safe ring buffer for accumulating audio samples.
///
/// This buffer is designed for use with real-time audio streaming where
/// samples are continuously appended and periodically consumed for processing.
actor AudioRingBuffer {
    /// The accumulated audio samples.
    private var samples: [Float] = []
    
    /// The total number of samples that have been processed and removed.
    private var processedSampleCount: Int = 0
    
    /// The sample rate of the audio (default: 16000 Hz).
    let sampleRate: Double
    
    /// Maximum buffer size in samples (default: 5 minutes at 16kHz).
    private let maxBufferSize: Int
    
    /// Creates a new audio ring buffer.
    /// - Parameters:
    ///   - sampleRate: The sample rate of the audio. Defaults to 16000 Hz.
    ///   - maxDurationSeconds: Maximum duration of audio to buffer. Defaults to 300 seconds (5 minutes).
    init(sampleRate: Double = 16000, maxDurationSeconds: Double = 300) {
        self.sampleRate = sampleRate
        self.maxBufferSize = Int(sampleRate * maxDurationSeconds)
    }
    
    /// Appends new samples to the buffer.
    /// - Parameter newSamples: The samples to append.
    /// - Note: If the buffer exceeds the maximum size, old samples are discarded.
    func append(_ newSamples: [Float]) {
        samples.append(contentsOf: newSamples)
        
        // Trim if exceeding max size (keep the most recent samples)
        if samples.count > maxBufferSize {
            let overflow = samples.count - maxBufferSize
            samples.removeFirst(overflow)
            processedSampleCount += overflow
        }
    }
    
    /// Returns all current samples without removing them.
    var currentSamples: [Float] {
        samples
    }
    
    /// Returns the number of samples currently in the buffer.
    var count: Int {
        samples.count
    }
    
    /// Returns the duration of audio in the buffer in seconds.
    var duration: TimeInterval {
        TimeInterval(samples.count) / sampleRate
    }
    
    /// Returns the total duration of audio processed (including consumed samples).
    var totalDuration: TimeInterval {
        TimeInterval(processedSampleCount + samples.count) / sampleRate
    }
    
    /// Consumes and removes samples from the beginning of the buffer.
    /// - Parameter count: The number of samples to consume.
    /// - Returns: The consumed samples.
    func consume(_ count: Int) -> [Float] {
        let consumeCount = min(count, samples.count)
        let consumed = Array(samples.prefix(consumeCount))
        samples.removeFirst(consumeCount)
        processedSampleCount += consumeCount
        return consumed
    }
    
    /// Consumes all samples from the buffer.
    /// - Returns: All samples in the buffer.
    func consumeAll() -> [Float] {
        let consumed = samples
        processedSampleCount += samples.count
        samples.removeAll()
        return consumed
    }
    
    /// Clears the buffer without updating the processed count.
    func clear() {
        samples.removeAll()
    }
    
    /// Resets the buffer completely, including the processed sample count.
    func reset() {
        samples.removeAll()
        processedSampleCount = 0
    }
    
    /// Returns samples from a specific offset without consuming them.
    /// - Parameters:
    ///   - offset: The offset in samples from the start of the current buffer.
    ///   - count: The number of samples to get.
    /// - Returns: The requested samples, or fewer if not enough are available.
    func getSamples(from offset: Int, count: Int) -> [Float] {
        guard offset >= 0 && offset < samples.count else { return [] }
        let endIndex = min(offset + count, samples.count)
        return Array(samples[offset..<endIndex])
    }
}
