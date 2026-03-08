use crate::llm::models::{Message, ToolDefinition};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// Delta event from a chat stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatDelta {
    /// Text content chunk
    Text(String),
    /// Thinking / Reasoning content chunk
    Thinking(String),
    /// Tool call chunk or event
    /// Note: For simplicity in the unification, we treat tool calls as atomic events here
    /// or simple chunks if the underlying provider streams them.
    /// For rig-core integration, we might receive fully formed tool calls in the stream.
    ToolCall {
        name: String,
        arguments: String,
        id: String,
    },
}

/// Unified trait for LLM clients
#[async_trait]
pub trait ChatClient: Send + Sync {
    /// Send a user message and get a response stream
    async fn chat_stream(
        &mut self,
        message: String,
        image_bytes: Option<Vec<u8>>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatDelta>> + Send>>>;

    /// Add an assistant message to history (used when consuming stream)
    fn add_assistant_message(&mut self, message: String) -> Result<()>;

    /// Add a generic message to history
    fn add_message(&mut self, message: Message) -> Result<()>;

    /// Set the system prompt (resets history)
    fn set_system_prompt(&mut self, prompt: String);

    /// Clear the conversation history
    fn clear_history(&mut self);

    /// Reset the session (for error recovery)
    async fn reset(&mut self) -> Result<()>;

    /// Set the tools definition for the client
    fn set_tools(&mut self, tools: Vec<ToolDefinition>);

    /// Set whether to enable the 'thinking' parameter
    fn set_enable_thinking(&mut self, enabled: bool);
}
