//! Response stream parser trait.
//!
//! `ResponseParser` converts raw SSE events from the API
//! into unified `MessageDelta` events that the agent loop consumes.

use crate::core::event::MessageDelta;
use anyhow::Result;

/// Parses SSE events from a specific API format into unified `MessageDelta`.
///
/// Each provider has its own SSE event structure:
/// - OpenAI Chat: `choices[0].delta.{content, tool_calls, ...}`
/// - OpenAI Responses: `response.output_text.delta`, `response.function_call_arguments.delta`, etc.
/// - Anthropic: `content_block_start`, `content_block_delta`, etc.
///
/// Implementations are **stateful** — some providers (like Anthropic) require
/// tracking active tool call IDs across events.
pub trait ResponseParser: Send {
    /// Parse a single SSE event. Returns `None` to skip/ignore the event.
    ///
    /// * `event_type` - The SSE event type (e.g., "message", "content_block_delta")
    /// * `data` - The raw JSON data string
    fn parse_sse(&mut self, event_type: &str, data: &str) -> Option<Result<MessageDelta>>;
}
