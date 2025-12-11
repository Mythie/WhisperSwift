import Foundation

/// The state of a streaming transcriber.
public enum StreamingState: Sendable, Equatable {
    /// The transcriber is initialized but not started.
    case idle
    
    /// The transcriber is actively processing audio.
    case running
    
    /// The transcriber is stopping and finalizing any remaining audio.
    case stopping
    
    /// The transcriber has stopped.
    case stopped
    
    /// The transcriber has encountered an error.
    case failed(WhisperError)
    
    /// A human-readable description of the state.
    public var description: String {
        switch self {
        case .idle: return "idle"
        case .running: return "running"
        case .stopping: return "stopping"
        case .stopped: return "stopped"
        case .failed(let error): return "failed(\(error.localizedDescription))"
        }
    }
    
    // Equatable conformance for associated value
    public static func == (lhs: StreamingState, rhs: StreamingState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle): return true
        case (.running, .running): return true
        case (.stopping, .stopping): return true
        case (.stopped, .stopped): return true
        case (.failed, .failed): return true
        default: return false
        }
    }
}
