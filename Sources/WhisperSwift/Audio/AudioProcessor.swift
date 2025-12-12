import Foundation
import AVFoundation

/// Utilities for processing audio into the format required by whisper.cpp.
///
/// whisper.cpp requires audio in the following format:
/// - Sample rate: 16000 Hz
/// - Channels: 1 (mono)
/// - Format: Float32 normalized to -1.0...1.0
public enum AudioProcessor {
    
    /// The required sample rate for whisper.cpp.
    public static let requiredSampleRate: Double = 16000
    
    /// Loads and converts an audio file to Float32 samples at 16kHz mono.
    /// - Parameter url: URL to the audio file.
    /// - Returns: Array of Float32 samples normalized to -1.0...1.0.
    /// - Throws: `WhisperError.audioLoadFailed` if the file cannot be loaded.
    public static func loadAudioFile(_ url: URL) throws -> [Float] {
        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(forReading: url)
        } catch {
            throw WhisperError.audioLoadFailed(url, underlying: error)
        }
        
        return try processAudioFile(audioFile)
    }
    
    /// Converts an AVAudioPCMBuffer to Float32 samples at 16kHz mono.
    /// - Parameters:
    ///   - buffer: The audio buffer to convert.
    ///   - sampleRate: The sample rate of the input buffer.
    /// - Returns: Array of Float32 samples normalized to -1.0...1.0.
    /// - Throws: `WhisperError.invalidAudioFormat` if conversion fails.
    public static func convert(_ buffer: AVAudioPCMBuffer, sampleRate: Double) throws -> [Float] {
        guard let floatData = buffer.floatChannelData else {
            throw WhisperError.invalidAudioFormat("Buffer does not contain float data")
        }
        
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        
        // Convert to mono if stereo
        var monoSamples: [Float]
        if channelCount == 1 {
            monoSamples = Array(UnsafeBufferPointer(start: floatData[0], count: frameCount))
        } else {
            // Average all channels to mono
            monoSamples = [Float](repeating: 0, count: frameCount)
            for channel in 0..<channelCount {
                let channelData = floatData[channel]
                for i in 0..<frameCount {
                    monoSamples[i] += channelData[i]
                }
            }
            let channelCountFloat = Float(channelCount)
            for i in 0..<frameCount {
                monoSamples[i] /= channelCountFloat
            }
        }
        
        // Resample to 16kHz if needed
        if abs(sampleRate - requiredSampleRate) > 1.0 {
            monoSamples = resample(monoSamples, from: sampleRate, to: requiredSampleRate)
        }
        
        return monoSamples
    }
    
    // MARK: - Private Helpers
    
    private static func processAudioFile(_ audioFile: AVAudioFile) throws -> [Float] {
        let inputFormat = audioFile.processingFormat
        let inputSampleRate = inputFormat.sampleRate
        let frameCount = AVAudioFrameCount(audioFile.length)
        
        // Create buffer for reading
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: frameCount
        ) else {
            throw WhisperError.invalidAudioFormat("Failed to create input buffer")
        }
        
        // Read the file
        do {
            try audioFile.read(into: inputBuffer)
        } catch {
            throw WhisperError.audioLoadFailed(audioFile.url, underlying: error)
        }
        
        // Convert to the format we need
        return try convert(inputBuffer, sampleRate: inputSampleRate)
    }
    
    // MARK: - Padding
    
    /// Minimum samples required for whisper.cpp (100ms at 16kHz).
    public static let minimumSamples = 1600
    
    /// Pads audio samples to meet the minimum length required by whisper.cpp.
    /// - Parameter samples: The input samples at 16kHz.
    /// - Returns: Samples padded with silence to at least 100ms, or original if already long enough.
    public static func padToMinimumLength(_ samples: [Float]) -> [Float] {
        guard samples.count < minimumSamples else { return samples }
        var padded = samples
        padded.append(contentsOf: [Float](repeating: 0, count: minimumSamples - samples.count))
        return padded
    }
    
    // MARK: - Resampling
    
    /// Simple linear interpolation resampling.
    /// For production, consider using vDSP for better quality.
    /// - Parameters:
    ///   - samples: The input samples.
    ///   - inputRate: The sample rate of the input.
    ///   - outputRate: The desired output sample rate.
    /// - Returns: Resampled audio samples.
    public static func resample(_ samples: [Float], from inputRate: Double, to outputRate: Double) -> [Float] {
        let ratio = inputRate / outputRate
        let outputLength = Int(Double(samples.count) / ratio)
        
        guard outputLength > 0 else { return [] }
        
        var output = [Float](repeating: 0, count: outputLength)
        
        for i in 0..<outputLength {
            let srcIndex = Double(i) * ratio
            let srcIndexInt = Int(srcIndex)
            let frac = Float(srcIndex - Double(srcIndexInt))
            
            if srcIndexInt + 1 < samples.count {
                // Linear interpolation
                output[i] = samples[srcIndexInt] * (1 - frac) + samples[srcIndexInt + 1] * frac
            } else if srcIndexInt < samples.count {
                output[i] = samples[srcIndexInt]
            }
        }
        
        return output
    }
}
