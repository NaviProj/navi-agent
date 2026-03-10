//! Unified API chat client that uses `MessageSerializer` + `ResponseParser`.
//!
//! This module provides `APIChatClient` which bridges the `LlmClient` trait
//! to HTTP-based LLM APIs. Format differences between providers are handled
//! by pluggable serializer/parser implementations rather than inline branching.

use crate::core::error::AgentError;
use crate::core::message::NaviMessage;
use crate::llm::anthropic::{AnthropicParser, AnthropicSerializer};
use crate::llm::openai_chat::{OpenAIChatParser, OpenAIChatSerializer};
use crate::llm::openai_responses::{OpenAIResponsesParser, OpenAIResponsesSerializer};
use crate::llm::parser::ResponseParser;
use crate::llm::serializer::{MessageSerializer, ToolDef};
use crate::runtime::llm_client::{LlmClient, LlmStream};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::Client;

/// Supported API providers / formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum APIFormat {
    OpenAIChat,
    OpenAIResponses,
    Anthropic,
}

/// Authentication method for the API.
#[derive(Clone)]
enum AuthMethod {
    Bearer(String),
    AnthropicKey(String),
}

/// Unified HTTP-based LLM client.
///
/// Uses `MessageSerializer` to build request payloads and `ResponseParser`
/// to parse SSE responses. The client itself is stateless — all conversation
/// state lives in the `NaviMessage` slice passed to each call.
#[derive(Clone)]
pub struct APIChatClient {
    client: Client,
    api_base: String,
    model_name: String,
    format: APIFormat,
    auth: AuthMethod,
    thinking: Option<bool>,
    max_tokens: Option<u32>,
}

impl APIChatClient {
    /// Create a new client.
    ///
    /// * `api_base` - Base URL. If `None`, defaults based on format.
    /// * `api_key` - API key for authentication.
    /// * `model` - Model identifier string.
    /// * `format` - Which API format to use.
    pub fn new(
        api_base: Option<String>,
        api_key: String,
        model: String,
        format: APIFormat,
    ) -> Self {
        Self::with_thinking(api_base, api_key, model, format, None)
    }

    pub fn with_thinking(
        api_base: Option<String>,
        api_key: String,
        model: String,
        format: APIFormat,
        thinking: Option<bool>,
    ) -> Self {
        let (default_base, auth) = match format {
            APIFormat::OpenAIChat => (
                "https://api.openai.com/v1/chat/completions".to_string(),
                AuthMethod::Bearer(api_key),
            ),
            APIFormat::OpenAIResponses => (
                "https://api.openai.com/v1/responses".to_string(),
                AuthMethod::Bearer(api_key),
            ),
            APIFormat::Anthropic => (
                "https://api.anthropic.com/v1/messages".to_string(),
                AuthMethod::AnthropicKey(api_key),
            ),
        };

        let mut api_base = api_base.unwrap_or(default_base);

        // Auto-append endpoint suffix if needed
        let suffix = match format {
            APIFormat::OpenAIChat => "chat/completions",
            APIFormat::OpenAIResponses => "responses",
            APIFormat::Anthropic => "messages",
        };
        if !api_base.ends_with(suffix) {
            if !api_base.ends_with('/') {
                api_base.push('/');
            }
            api_base.push_str(suffix);
        }

        // Default to enabled for kimi-k2.5 if not specified
        let thinking = thinking.or_else(|| {
            if model.contains("kimi-k2.5") {
                Some(true)
            } else {
                None
            }
        });

        Self {
            client: Client::new(),
            api_base,
            model_name: model,
            format,
            auth,
            thinking,
            max_tokens: None,
        }
    }

    /// Set whether to enable the 'thinking' parameter
    pub fn set_enable_thinking(&mut self, enabled: bool) {
        self.thinking = Some(enabled);
    }

    /// Set max_tokens for the LLM response. Particularly important for Anthropic
    /// which requires it in the request body.
    pub fn set_max_tokens(&mut self, max_tokens: u32) {
        self.max_tokens = Some(max_tokens);
    }

    /// Detect format from api_base / model name (backward compat helper).
    pub fn detect(
        api_base: Option<String>,
        api_key: String,
        model: String,
        use_responses_api: bool,
    ) -> Self {
        Self::detect_with_thinking(api_base, api_key, model, use_responses_api, None)
    }

    pub fn detect_with_thinking(
        api_base: Option<String>,
        api_key: String,
        model: String,
        use_responses_api: bool,
        thinking: Option<bool>,
    ) -> Self {
        let format = if api_base.as_deref().unwrap_or("").contains("anthropic")
            || api_base.as_deref().unwrap_or("").ends_with("messages")
            || model.contains("claude")
        {
            APIFormat::Anthropic
        } else if use_responses_api {
            APIFormat::OpenAIResponses
        } else {
            APIFormat::OpenAIChat
        };

        Self::with_thinking(api_base, api_key, model, format, thinking)
    }

    fn serializer(&self) -> Box<dyn MessageSerializer> {
        match self.format {
            APIFormat::OpenAIChat => Box::new(OpenAIChatSerializer),
            APIFormat::OpenAIResponses => Box::new(OpenAIResponsesSerializer),
            APIFormat::Anthropic => {
                let mut s = AnthropicSerializer::default();
                if let Some(mt) = self.max_tokens {
                    s.max_tokens = mt;
                }
                Box::new(s)
            }
        }
    }

    fn parser(&self) -> Box<dyn ResponseParser> {
        match self.format {
            APIFormat::OpenAIChat => Box::new(OpenAIChatParser::new()),
            APIFormat::OpenAIResponses => Box::new(OpenAIResponsesParser::new()),
            APIFormat::Anthropic => Box::new(AnthropicParser::new()),
        }
    }
}

#[async_trait]
impl LlmClient for APIChatClient {
    async fn stream_completion(
        &self,
        system_prompt: &str,
        messages: &[NaviMessage],
        tools: &[ToolDef],
    ) -> Result<LlmStream, AgentError> {
        let sp = if system_prompt.is_empty() {
            None
        } else {
            Some(system_prompt)
        };

        let payload = self.serializer().serialize_payload(
            &self.model_name,
            sp,
            messages,
            tools,
            true,
            self.thinking,
        );

        tracing::info!(
            "LLM Request payload: {}",
            serde_json::to_string(&payload).unwrap_or_default()
        );

        let mut req_builder = self.client.post(&self.api_base);

        match &self.auth {
            AuthMethod::Bearer(key) => {
                req_builder = req_builder.bearer_auth(key);
            }
            AuthMethod::AnthropicKey(key) => {
                req_builder = req_builder
                    .header("x-api-key", key)
                    .header("anthropic-version", "2023-06-01");
            }
        }

        let response = req_builder
            .json(&payload)
            .send()
            .await
            .map_err(|e| AgentError::LlmError(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            tracing::error!("API Error {}: {}", status, text);
            return Err(AgentError::LlmError(format!(
                "API Error {}: {}",
                status, text
            )));
        }

        let mut parser = self.parser();

        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(move |event_result| {
                let result = match event_result {
                    Ok(event) => {
                        tracing::debug!(
                            "LLM Response chunk: event='{}' data='{}'",
                            event.event,
                            event.data
                        );
                        parser.parse_sse(&event.event, &event.data)
                    }
                    Err(e) => {
                        tracing::error!("SSE Error: {}", e);
                        Some(Err(anyhow::anyhow!("SSE Error: {}", e)))
                    }
                };
                std::future::ready(result)
            })
            .map(|r| r.map_err(|e| AgentError::LlmError(e.to_string())));

        Ok(Box::pin(stream))
    }
}
