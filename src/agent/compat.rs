//! Backward-compatible adapter: `NaviBotLlmClient` wraps `APIChatClient`.
//!
//! This provides the same constructor API that callers (nanocode, schedule_agent, etc.)
//! are already using, but delegates to the refactored `APIChatClient` which now
//! directly implements `LlmClient`.

use crate::llm::api_client::APIChatClient;
use crate::runtime::llm_client::LlmClient;

/// A convenience wrapper around `APIChatClient` that provides the old constructor API.
///
/// New code should prefer constructing `APIChatClient::new()` or `APIChatClient::detect()` directly.
pub struct NaviBotLlmClient {
    inner: APIChatClient,
}

impl NaviBotLlmClient {
    pub fn new(
        api_base: Option<String>,
        api_key: String,
        model_name: impl Into<String>,
        use_responses_api: bool,
        enable_thinking: bool,
    ) -> Self {
        Self {
            inner: APIChatClient::detect_with_thinking(
                api_base,
                api_key,
                model_name.into(),
                use_responses_api,
                Some(enable_thinking),
            ),
        }
    }
}

// Delegate LlmClient to the inner APIChatClient
use crate::core::error::AgentError;
use crate::core::message::NaviMessage;
use crate::llm::serializer::ToolDef;
use crate::runtime::llm_client::LlmStream;
use async_trait::async_trait;

#[async_trait]
impl LlmClient for NaviBotLlmClient {
    async fn stream_completion(
        &self,
        system_prompt: &str,
        messages: &[NaviMessage],
        tools: &[ToolDef],
    ) -> Result<LlmStream, AgentError> {
        self.inner
            .stream_completion(system_prompt, messages, tools)
            .await
    }
}
