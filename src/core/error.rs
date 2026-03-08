use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Agent is already streaming")]
    AlreadyStreaming,
    #[error("No model configured")]
    NoModel,
    #[error("No messages to continue from")]
    NoMessages,
    #[error("Cannot continue from assistant message")]
    InvalidContinuation,
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Tool execution failed: {0}")]
    ToolExecutionFailed(String),
    #[error("LLM error: {0}")]
    LlmError(String),
    #[error("Aborted")]
    Aborted,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
