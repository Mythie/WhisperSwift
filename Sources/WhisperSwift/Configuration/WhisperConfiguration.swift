import Foundation

/// Hardware and processing configuration for whisper.cpp.
public struct WhisperConfiguration: Sendable {
    /// Whether to use GPU acceleration (Metal on macOS).
    ///
    /// Automatically disabled on iOS simulator. On real devices,
    /// this provides significant performance improvements.
    public var useGPU: Bool
    
    /// Whether to use flash attention for better Metal performance.
    ///
    /// Recommended when `useGPU` is true on Apple Silicon.
    public var useFlashAttention: Bool
    
    /// Number of CPU threads to use.
    ///
    /// If `nil`, automatically determined based on processor count.
    /// When using GPU, fewer CPU threads are typically needed.
    public var threadCount: Int?
    
    /// Creates a new configuration with the specified options.
    /// - Parameters:
    ///   - useGPU: Whether to use GPU acceleration. Defaults to `true`.
    ///   - useFlashAttention: Whether to use flash attention. Defaults to `true`.
    ///   - threadCount: Number of CPU threads. Defaults to `nil` (auto).
    public init(
        useGPU: Bool = true,
        useFlashAttention: Bool = true,
        threadCount: Int? = nil
    ) {
        self.useGPU = useGPU
        self.useFlashAttention = useFlashAttention
        self.threadCount = threadCount
    }
    
    /// Default configuration with GPU acceleration enabled.
    public static let `default` = WhisperConfiguration()
    
    /// CPU-only configuration (useful for testing or fallback).
    public static let cpuOnly = WhisperConfiguration(
        useGPU: false,
        useFlashAttention: false
    )
    
    /// Returns the optimal thread count based on the current configuration.
    internal var optimalThreadCount: Int32 {
        if let count = threadCount {
            return Int32(max(1, count))
        }
        // Leave 2 cores free for system tasks, cap at 8
        let cpuCount = ProcessInfo.processInfo.processorCount
        return Int32(max(1, min(8, cpuCount - 2)))
    }
}
