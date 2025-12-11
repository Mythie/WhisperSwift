import Foundation
import Accelerate

/// Configuration for silence detection.
public struct SilenceDetectorOptions: Sendable {
    /// RMS threshold below which audio is considered silence (0.0-1.0).
    ///
    /// Lower values are more sensitive to quiet sounds.
    /// - Default: 0.01
    public var threshold: Float
    
    /// Minimum duration of silence in seconds to consider it a valid break point.
    /// - Default: 0.3 seconds (300ms)
    public var minSilenceDuration: TimeInterval
    
    /// How far back from the end of audio to search for silence, in seconds.
    /// - Default: 5.0 seconds
    public var searchDuration: TimeInterval
    
    /// Window size in seconds for RMS calculation.
    /// - Default: 0.01 seconds (10ms)
    public var windowDuration: TimeInterval
    
    /// Creates silence detector options.
    /// - Parameters:
    ///   - threshold: RMS threshold for silence. Defaults to 0.01.
    ///   - minSilenceDuration: Minimum silence duration. Defaults to 0.3s.
    ///   - searchDuration: How far back to search. Defaults to 5.0s.
    ///   - windowDuration: RMS window size. Defaults to 0.01s.
    public init(
        threshold: Float = 0.01,
        minSilenceDuration: TimeInterval = 0.3,
        searchDuration: TimeInterval = 5.0,
        windowDuration: TimeInterval = 0.01
    ) {
        self.threshold = threshold
        self.minSilenceDuration = minSilenceDuration
        self.searchDuration = searchDuration
        self.windowDuration = windowDuration
    }
    
    /// Default silence detector options.
    public static let `default` = SilenceDetectorOptions()
}

/// A detected speech segment based on silence detection.
public struct SpeechSegment: Sendable {
    /// Start sample index in the original audio.
    public let startSample: Int
    /// End sample index in the original audio.
    public let endSample: Int
    
    /// Start time in seconds (assuming 16kHz sample rate).
    public var startTime: TimeInterval {
        TimeInterval(startSample) / 16000.0
    }
    
    /// End time in seconds (assuming 16kHz sample rate).
    public var endTime: TimeInterval {
        TimeInterval(endSample) / 16000.0
    }
    
    /// Duration in seconds.
    public var duration: TimeInterval {
        endTime - startTime
    }
}

/// Lightweight silence detection using RMS (Root Mean Square) analysis.
///
/// This provides a simple, model-free approach to detecting speech boundaries
/// by analyzing audio energy levels. It's much lighter weight than neural
/// network-based VAD models.
///
/// ```swift
/// let detector = SilenceDetector()
/// if let breakPoint = detector.findSilenceBreak(in: samples) {
///     // Split audio at breakPoint for transcription
///     let chunk = Array(samples[..<breakPoint])
///     let remaining = Array(samples[breakPoint...])
/// }
/// ```
public enum SilenceDetector {
    
    /// Sample rate expected for input audio.
    public static let sampleRate: Double = 16000
    
    /// Finds the best silence break point near the end of audio.
    ///
    /// Searches backwards from the end of the audio to find a silence gap
    /// that can be used as a natural split point for chunked transcription.
    ///
    /// - Parameters:
    ///   - samples: Audio samples at 16kHz.
    ///   - options: Silence detection options.
    /// - Returns: Sample index of the start of the silence gap (where speech ended),
    ///            or nil if no suitable silence was found.
    public static func findSilenceBreak(
        in samples: [Float],
        options: SilenceDetectorOptions = .default
    ) -> Int? {
        let windowSize = Int(options.windowDuration * sampleRate)
        let searchSamples = Int(options.searchDuration * sampleRate)
        let minSilenceSamples = Int(options.minSilenceDuration * sampleRate)
        
        guard samples.count > windowSize else { return nil }
        
        let searchStart = max(0, samples.count - searchSamples)
        
        var bestSilenceStart: Int? = nil
        var bestSilenceLength = 0
        
        var currentSilenceStart: Int? = nil
        var currentSilenceLength = 0
        
        // Search backwards from the end
        var i = samples.count - windowSize
        while i >= searchStart {
            let rms = calculateRMS(samples: samples, start: i, count: windowSize)
            
            if rms < options.threshold {
                // In silence
                if currentSilenceStart == nil {
                    currentSilenceStart = i + windowSize // End of silence region
                }
                currentSilenceLength += windowSize
            } else {
                // Not silence - check if we found a good gap
                if let silenceStart = currentSilenceStart,
                   currentSilenceLength >= minSilenceSamples,
                   currentSilenceLength > bestSilenceLength {
                    bestSilenceStart = silenceStart
                    bestSilenceLength = currentSilenceLength
                }
                currentSilenceStart = nil
                currentSilenceLength = 0
            }
            
            i -= windowSize
        }
        
        // Check final run
        if let silenceStart = currentSilenceStart,
           currentSilenceLength >= minSilenceSamples,
           currentSilenceLength > bestSilenceLength {
            bestSilenceStart = silenceStart
            bestSilenceLength = currentSilenceLength
        }
        
        // Return the start of the silence (where speech ended)
        if let silenceStart = bestSilenceStart {
            return silenceStart - bestSilenceLength
        }
        
        return nil
    }
    
    /// Detects whether the audio contains speech (non-silence).
    ///
    /// - Parameters:
    ///   - samples: Audio samples at 16kHz.
    ///   - threshold: RMS threshold for silence. Defaults to 0.01.
    /// - Returns: `true` if speech (audio above threshold) is detected.
    public static func containsSpeech(
        in samples: [Float],
        threshold: Float = 0.01
    ) -> Bool {
        guard !samples.isEmpty else { return false }
        
        let rms = calculateRMS(samples: samples, start: 0, count: samples.count)
        return rms >= threshold
    }
    
    /// Calculates the RMS (Root Mean Square) energy of audio samples.
    ///
    /// - Parameters:
    ///   - samples: Audio samples.
    ///   - start: Start index.
    ///   - count: Number of samples to analyze.
    /// - Returns: RMS value (0.0 to ~1.0 for normalized audio).
    public static func calculateRMS(
        samples: [Float],
        start: Int,
        count: Int
    ) -> Float {
        let actualCount = min(count, samples.count - start)
        guard actualCount > 0, start >= 0, start < samples.count else { return 0 }
        
        // Use Accelerate for efficient computation
        var sumOfSquares: Float = 0
        samples.withUnsafeBufferPointer { buffer in
            let ptr = buffer.baseAddress! + start
            vDSP_svesq(ptr, 1, &sumOfSquares, vDSP_Length(actualCount))
        }
        
        let meanSquare = sumOfSquares / Float(actualCount)
        return sqrtf(meanSquare)
    }
    
    /// Finds all speech segments in audio by detecting silence gaps.
    ///
    /// - Parameters:
    ///   - samples: Audio samples at 16kHz.
    ///   - options: Silence detection options.
    /// - Returns: Array of speech segments with start/end sample indices.
    public static func findSpeechSegments(
        in samples: [Float],
        options: SilenceDetectorOptions = .default
    ) -> [SpeechSegment] {
        let windowSize = Int(options.windowDuration * sampleRate)
        let minSilenceSamples = Int(options.minSilenceDuration * sampleRate)
        
        guard samples.count > windowSize else {
            // Return entire audio as one segment if too short
            if !samples.isEmpty && containsSpeech(in: samples, threshold: options.threshold) {
                return [SpeechSegment(startSample: 0, endSample: samples.count)]
            }
            return []
        }
        
        var segments: [SpeechSegment] = []
        var speechStart: Int? = nil
        var silenceLength = 0
        
        var i = 0
        while i < samples.count - windowSize {
            let rms = calculateRMS(samples: samples, start: i, count: windowSize)
            
            if rms >= options.threshold {
                // Speech detected
                if speechStart == nil {
                    speechStart = i
                }
                silenceLength = 0
            } else {
                // Silence
                silenceLength += windowSize
                
                // If we've had enough silence, end the current speech segment
                if let start = speechStart, silenceLength >= minSilenceSamples {
                    let endSample = i - silenceLength + windowSize
                    if endSample > start {
                        segments.append(SpeechSegment(startSample: start, endSample: endSample))
                    }
                    speechStart = nil
                }
            }
            
            i += windowSize
        }
        
        // Handle final segment
        if let start = speechStart {
            segments.append(SpeechSegment(startSample: start, endSample: samples.count))
        }
        
        return segments
    }
}
