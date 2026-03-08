//! Stateful chat client wrapper.
//!
//! Implements the legacy `ChatClient` trait on top of the new stateless `APIChatClient`.
//! This maintains conversation history internally and provides the `.chat_stream()`,
//! `.add_assistant_message()`, `.reset()` etc. API that `navi-bot::Cortex` and
//! `navi-server::SemanticDecider` depend on.

use crate::core::message::NaviMessage;
use crate::llm::api_client::APIChatClient;
use crate::llm::models::{Message, ToolDefinition};
use crate::llm::serializer::ToolDef;
use crate::llm::traits::{ChatClient, ChatDelta};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;

/// A stateful chat client that wraps the new stateless `APIChatClient`.
///
/// It manages conversation history and converts between the old `ChatClient`
/// API (used by Cortex, SemanticDecider) and the new `LlmClient` / serializer
/// architecture.
pub struct StatefulChatClient {
    inner: APIChatClient,
    history: Vec<NaviMessage>,
    system_prompt: Option<String>,
    tools: Vec<ToolDef>,
}

impl StatefulChatClient {
    /// Create a new stateful wrapper.
    pub fn new(
        api_base: Option<String>,
        api_key: String,
        model: String,
        use_responses_api: bool,
        enable_thinking: bool,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            inner: APIChatClient::detect_with_thinking(
                api_base,
                api_key,
                model,
                use_responses_api,
                Some(enable_thinking),
            ),
            history: Vec::new(),
            system_prompt,
            tools: Vec::new(),
        }
    }
}

#[async_trait]
impl ChatClient for StatefulChatClient {
    async fn chat_stream(
        &mut self,
        message: String,
        _image_bytes: Option<Vec<u8>>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatDelta>> + Send>>> {
        // Add user message to history
        self.history.push(NaviMessage::new_user_text(&message));

        let sp = self.system_prompt.as_deref().unwrap_or("");

        // Use the stateless inner client
        use crate::runtime::llm_client::LlmClient;
        let stream = self
            .inner
            .stream_completion(sp, &self.history, &self.tools)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Map MessageDelta -> ChatDelta
        let mapped = stream.map(|r| {
            r.map(|delta| {
                use crate::core::event::MessageDelta;
                match delta {
                    MessageDelta::Text(t) => ChatDelta::Text(t),
                    MessageDelta::Thinking(t) => ChatDelta::Thinking(t),
                    MessageDelta::ToolCall {
                        id,
                        name,
                        arguments_delta,
                    } => ChatDelta::ToolCall {
                        name,
                        arguments: arguments_delta,
                        id,
                    },
                }
            })
            .map_err(|e| anyhow::anyhow!("{}", e))
        });

        Ok(Box::pin(mapped))
    }

    fn add_assistant_message(&mut self, message: String) -> Result<()> {
        self.history
            .push(NaviMessage::new_assistant_text(message, None));
        Ok(())
    }

    fn add_message(&mut self, _message: Message) -> Result<()> {
        // Legacy compat — not used in practice by Cortex
        Ok(())
    }

    fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Some(prompt);
        self.history.clear();
    }

    fn clear_history(&mut self) {
        self.history.clear();
    }

    async fn reset(&mut self) -> Result<()> {
        self.history.clear();
        Ok(())
    }

    fn set_tools(&mut self, tools: Vec<ToolDefinition>) {
        self.tools = tools
            .into_iter()
            .map(|t| ToolDef {
                name: t.function.name,
                description: t.function.description,
                parameters: t.function.parameters,
            })
            .collect();
    }

    fn set_enable_thinking(&mut self, enabled: bool) {
        self.inner.set_enable_thinking(enabled);
    }
}
