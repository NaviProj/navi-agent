//! Message serialization trait and common types.
//!
//! `MessageSerializer` converts `NaviMessage` history directly into
//! the JSON payload expected by a specific API provider. This is the
//! single place where format differences between providers are handled.

use crate::core::message::NaviMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Provider-agnostic tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Serializes `NaviMessage` history into an API-specific JSON request body.
///
/// Implementations exist for each supported provider format:
/// - OpenAI Chat Completions (`messages` array)
/// - OpenAI Responses API (`input` array with typed items)
/// - Anthropic Messages API (alternating roles, `system` top-level)
pub trait MessageSerializer: Send + Sync {
    /// Build the full HTTP request body for a streaming completion.
    fn serialize_payload(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        messages: &[NaviMessage],
        tools: &[ToolDef],
        stream: bool,
        thinking: Option<bool>,
    ) -> Value;
}
