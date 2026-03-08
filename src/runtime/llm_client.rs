use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use crate::core::error::AgentError;
use crate::core::event::MessageDelta;
use crate::core::message::NaviMessage;
use crate::llm::serializer::ToolDef;

/// A stream of deltas from the LLM.
pub type LlmStream = Pin<Box<dyn Stream<Item = Result<MessageDelta, AgentError>> + Send>>;

/// Abstraction over the LLM provider.
///
/// This trait decouples `agent_loop` from any specific LLM SDK.
/// Implementations are responsible for:
/// 1. Accepting a system prompt, chat history (as `NaviMessage`), and tool definitions.
/// 2. Serializing them into the provider-specific format.
/// 3. Returning a stream of `MessageDelta` items.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Start a streaming completion with the given context.
    ///
    /// * `system_prompt` - The system prompt for the agent.
    /// * `messages` - The conversation history as `NaviMessage` (provider-agnostic).
    /// * `tools` - Tool definitions available to the model.
    async fn stream_completion(
        &self,
        system_prompt: &str,
        messages: &[NaviMessage],
        tools: &[ToolDef],
    ) -> Result<LlmStream, AgentError>;
}
