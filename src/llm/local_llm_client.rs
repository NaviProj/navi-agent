//! Local LLM client implementing the `LlmClient` trait for the agent framework.
//!
//! This bridges `navi-llm` session-based inference with the agent's `LlmClient` abstraction.
//! Uses a dedicated worker thread for llama.cpp inference (which is not Send),
//! communicating via channels.

use crate::core::error::AgentError;
use crate::core::event::MessageDelta;
use crate::core::message::{ContentBlock, MessageRole, NaviMessage};
use crate::llm::serializer::ToolDef;
use crate::runtime::llm_client::{LlmClient, LlmStream};
use async_trait::async_trait;
use navi_llm::{LlmConfig, LlmSessionFactory};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Local LLM client that implements `LlmClient` for use with `AgentBuilder`.
///
/// Unlike `LocalLMClient` (which implements the legacy `ChatClient` trait),
/// this directly implements the stateless `LlmClient` trait used by the agent loop.
pub struct LocalLlmAgentClient {
    factory: Arc<LlmSessionFactory>,
    enable_thinking: bool,
    caller_managed_context: bool,
    /// Optional overrides for session creation (allows sharing a factory with different session params)
    ctx_size_override: Option<u32>,
    max_tokens_override: Option<u32>,
}

#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl LocalLlmAgentClient {
    pub fn new(config: LlmConfig) -> Result<Self, anyhow::Error> {
        Self::new_with_context_mode(config, true)
    }

    pub fn new_with_context_mode(
        config: LlmConfig,
        caller_managed_context: bool,
    ) -> Result<Self, anyhow::Error> {
        let enable_thinking = config.enable_thinking;
        let factory = Arc::new(LlmSessionFactory::new(config)?);
        Ok(Self {
            factory,
            enable_thinking,
            caller_managed_context,
            ctx_size_override: None,
            max_tokens_override: None,
        })
    }

    pub fn from_factory(
        factory: Arc<LlmSessionFactory>,
        enable_thinking: bool,
        caller_managed_context: bool,
    ) -> Self {
        Self {
            factory,
            enable_thinking,
            caller_managed_context,
            ctx_size_override: None,
            max_tokens_override: None,
        }
    }

    /// Create from a shared factory with custom session parameters.
    ///
    /// This allows reusing a loaded model while overriding ctx_size and max_tokens
    /// for sessions created by this client.
    pub fn from_factory_with_options(
        factory: Arc<LlmSessionFactory>,
        enable_thinking: bool,
        caller_managed_context: bool,
        ctx_size: Option<u32>,
        max_tokens: Option<u32>,
    ) -> Self {
        Self {
            factory,
            enable_thinking,
            caller_managed_context,
            ctx_size_override: ctx_size,
            max_tokens_override: max_tokens,
        }
    }
}

/// Convert NaviMessage to navi_llm::ChatMessage.
fn to_chat_message(msg: &NaviMessage) -> Vec<navi_llm::ChatMessage> {
    if let NaviMessage::LLM(llm_msg) = msg {
        let mut full_content = String::new();
        let mut tool_calls = Vec::new();
        let mut tool_call_id = None;

        for block in &llm_msg.content {
            match block {
                ContentBlock::Text(t) => {
                    if !full_content.is_empty() {
                        full_content.push('\n');
                    }
                    full_content.push_str(t);
                }
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    let arguments_str = if arguments.is_string() {
                        arguments.as_str().unwrap_or_default().to_string()
                    } else {
                        arguments.to_string()
                    };
                    tool_calls.push(serde_json::json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments_str
                        }
                    }));
                }
                ContentBlock::ToolResult {
                    tool_call_id: tid,
                    content,
                    is_error,
                    ..
                } => {
                    if !full_content.is_empty() {
                        full_content.push('\n');
                    }
                    if *is_error {
                        full_content.push_str(&format!("Error: {}", content));
                    } else {
                        full_content.push_str(content);
                    }
                    tool_call_id = Some(tid.clone());
                }
                ContentBlock::Thinking(t) => {
                    if !full_content.is_empty() {
                        full_content.push('\n');
                    }
                    full_content.push_str(&format!("<think>\n{}\n</think>", t));
                }
            }
        }

        let mut chat_msg = match llm_msg.role {
            MessageRole::System => navi_llm::ChatMessage::system(full_content),
            MessageRole::User => navi_llm::ChatMessage::user(full_content),
            MessageRole::Assistant => navi_llm::ChatMessage::assistant(full_content),
            MessageRole::ToolResult => {
                navi_llm::ChatMessage::tool(full_content, tool_call_id.unwrap_or_default())
            }
        };

        if !tool_calls.is_empty() {
            chat_msg.tool_calls = Some(serde_json::Value::Array(tool_calls));
        }

        vec![chat_msg]
    } else {
        Vec::new()
    }
}

#[async_trait]
impl LlmClient for LocalLlmAgentClient {
    async fn stream_completion(
        &self,
        system_prompt: &str,
        messages: &[NaviMessage],
        tools: &[ToolDef],
    ) -> Result<LlmStream, AgentError> {
        let tools_json = if !tools.is_empty() {
            let wrapped_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            Some(
                serde_json::to_string(&wrapped_tools)
                    .map_err(|e| AgentError::LlmError(e.to_string()))?,
            )
        } else {
            None
        };
        let factory = self.factory.clone();
        let enable_thinking = self.enable_thinking;
        let caller_managed_context = self.caller_managed_context;
        let ctx_size_override = self.ctx_size_override;
        let max_tokens_override = self.max_tokens_override;

        let mut chat_history = Vec::new();
        for msg in messages {
            chat_history.extend(to_chat_message(msg));
        }
        let system = system_prompt.to_string();

        if chat_history.is_empty() && system.is_empty() {
            return Err(AgentError::LlmError(
                "No prompt or history provided".to_string(),
            ));
        }

        let (delta_tx, mut delta_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<MessageDelta, AgentError>>();

        tokio::task::spawn_blocking(move || {
            let cfg = factory.config();
            let ctx_size = ctx_size_override.unwrap_or(cfg.ctx_size.get());
            let max_tokens = max_tokens_override.unwrap_or(cfg.max_tokens);
            let mut session = match factory.create_session_with_options(
                Some(ctx_size),
                Some(max_tokens),
            ) {
                Ok(s) => s,
                Err(e) => {
                    let _ = delta_tx.send(Err(AgentError::LlmError(format!(
                        "Failed to create LLM session: {}",
                        e
                    ))));
                    return;
                }
            };

            if caller_managed_context {
                session.clear();
                let mut full_history = Vec::new();
                if !system.is_empty() {
                    full_history.push(navi_llm::ChatMessage::system(&system));
                }
                full_history.extend(chat_history);
                if !full_history.is_empty() {
                    session.set_messages(full_history);
                }
            } else if let Some(last) = chat_history.last().cloned() {
                session.clear();
                if !system.is_empty() {
                    session.set_system_prompt(&system);
                }
                session.add_message(last);
            }

            if let Some(tj) = tools_json {
                session.set_tools_json(Some(tj));
            }

            // Set up external cancel flag so generation can be stopped
            // when the stream consumer (agent loop) is dropped.
            let cancel_flag = Arc::new(AtomicBool::new(false));
            session.set_cancel(Some(cancel_flag.clone()));

            let mut in_think = false;
            let mut buffer = String::new();
            let delta_tx_clone = delta_tx.clone();
            let mut fallback_tool_id: u64 = 0;
            let mut pending_tool_calls: BTreeMap<usize, PendingToolCall> = BTreeMap::new();

            let flush_buffer = |in_think: &mut bool,
                                buffer: &mut String,
                                is_end: bool,
                                tx: &tokio::sync::mpsc::UnboundedSender<
                Result<MessageDelta, AgentError>,
            >| {
                loop {
                    if *in_think {
                        if let Some(pos) = buffer.find("</think>") {
                            let before = &buffer[..pos];
                            if enable_thinking && !before.is_empty() {
                                let _ = tx.send(Ok(MessageDelta::Thinking(before.to_string())));
                            }
                            *in_think = false;
                            let mut after = buffer[pos + 8..].to_string();
                            if after.starts_with('\n') {
                                after = after[1..].to_string();
                            }
                            *buffer = after;
                        } else if is_end {
                            if enable_thinking && !buffer.is_empty() {
                                let _ = tx.send(Ok(MessageDelta::Thinking(buffer.to_string())));
                            }
                            buffer.clear();
                            break;
                        } else {
                            let mut safe_len = buffer.len().saturating_sub(8);
                            while safe_len > 0 && !buffer.is_char_boundary(safe_len) {
                                safe_len -= 1;
                            }
                            if safe_len > 0 {
                                let chunk = buffer[..safe_len].to_string();
                                if enable_thinking {
                                    let _ = tx.send(Ok(MessageDelta::Thinking(chunk)));
                                }
                                buffer.drain(..safe_len);
                            }
                            break;
                        }
                    } else {
                        if let Some(pos) = buffer.find("<think>") {
                            let before = &buffer[..pos];
                            if !before.is_empty() {
                                let _ = tx.send(Ok(MessageDelta::Text(before.to_string())));
                            }
                            *in_think = true;
                            let mut after = buffer[pos + 7..].to_string();
                            if after.starts_with('\n') {
                                after = after[1..].to_string();
                            }
                            *buffer = after;
                        } else if is_end {
                            if !buffer.is_empty() {
                                let _ = tx.send(Ok(MessageDelta::Text(buffer.to_string())));
                            }
                            buffer.clear();
                            break;
                        } else {
                            let mut safe_len = buffer.len().saturating_sub(7);
                            while safe_len > 0 && !buffer.is_char_boundary(safe_len) {
                                safe_len -= 1;
                            }
                            if safe_len > 0 {
                                let chunk = buffer[..safe_len].to_string();
                                let _ = tx.send(Ok(MessageDelta::Text(chunk)));
                                buffer.drain(..safe_len);
                            }
                            break;
                        }
                    }
                }
            };

            let send_delta_from_json = |val: &Value,
                                        fallback_tool_id: &mut u64,
                                        pending_tool_calls: &mut BTreeMap<usize, PendingToolCall>,
                                        tx: &tokio::sync::mpsc::UnboundedSender<
                Result<MessageDelta, AgentError>,
            >| {
                // Try two formats:
                // 1. SSE wrapped: {"choices":[{"delta":{"tool_calls":[...]}}]}
                // 2. Direct from OAI parser: {"tool_calls":[...], "content":"...", ...}
                let delta = if let Some(d) = val
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                {
                    std::borrow::Cow::Borrowed(d)
                } else if val.get("tool_calls").is_some()
                    || val.get("content").is_some()
                    || val.get("reasoning_content").is_some()
                {
                    // Direct format — treat the root object as the delta
                    std::borrow::Cow::Borrowed(val)
                } else {
                    return false;
                };

                if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for (array_index, tc) in tool_calls.iter().enumerate() {
                        let index = tc
                            .get("index")
                            .and_then(|i| i.as_u64())
                            .map(|i| i as usize)
                            .unwrap_or(array_index);
                        let entry = pending_tool_calls.entry(index).or_default();

                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        if !name.is_empty() {
                            entry.name = name;
                        }

                        let args = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .map(|a| {
                                if let Some(s) = a.as_str() {
                                    s.to_string()
                                } else {
                                    a.to_string()
                                }
                            })
                            .unwrap_or_default();
                        if !args.is_empty() {
                            entry.arguments.push_str(&args);
                        }

                        let mut id = tc
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or("")
                            .to_string();
                        if id.is_empty() {
                            id = entry.id.clone();
                        }
                        if id.is_empty() && (!entry.name.is_empty() || !entry.arguments.is_empty()) {
                            *fallback_tool_id += 1;
                            id = format!("local-call-{}", *fallback_tool_id);
                        }
                        if !id.is_empty() {
                            entry.id = id;
                        }
                    }
                }

                if let Some(thinking) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                    if !thinking.is_empty() {
                        let _ = tx.send(Ok(MessageDelta::Thinking(thinking.to_string())));
                    }
                }

                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        let _ = tx.send(Ok(MessageDelta::Text(content.to_string())));
                    }
                }

                true
            };

            let result = session.complete_chat_streaming(|chunk| {
                // If the stream consumer (agent loop) has been dropped, signal cancel
                if delta_tx_clone.is_closed() {
                    cancel_flag.store(true, Ordering::Relaxed);
                    return;
                }

                let trimmed = chunk.trim_start();
                if trimmed.starts_with('{') {
                    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                        if send_delta_from_json(
                            &val,
                            &mut fallback_tool_id,
                            &mut pending_tool_calls,
                            &delta_tx_clone,
                        ) {
                            return;
                        }
                    }
                }

                if let Ok(val) = serde_json::from_str::<Value>(chunk) {
                    if send_delta_from_json(
                        &val,
                        &mut fallback_tool_id,
                        &mut pending_tool_calls,
                        &delta_tx_clone,
                    ) {
                        return;
                    }
                }

                if let Some(data) = chunk.strip_prefix("data: ").map(str::trim) {
                    if let Ok(val) = serde_json::from_str::<Value>(data) {
                        if send_delta_from_json(
                            &val,
                            &mut fallback_tool_id,
                            &mut pending_tool_calls,
                            &delta_tx_clone,
                        ) {
                            return;
                        }
                    }
                }

                buffer.push_str(chunk);
                flush_buffer(&mut in_think, &mut buffer, false, &delta_tx_clone);
            });

            if result.is_ok() {
                tracing::debug!(
                    "Local LLM generation complete, pending_tool_calls count: {}",
                    pending_tool_calls.len()
                );
                for (idx, mut tc) in pending_tool_calls {
                    if tc.id.is_empty() && (!tc.name.is_empty() || !tc.arguments.is_empty()) {
                        fallback_tool_id += 1;
                        tc.id = format!("local-call-{}", fallback_tool_id);
                    }
                    if !tc.id.is_empty() && !tc.name.is_empty() {
                        tracing::info!(
                            "Local LLM tool call [{}]: name={}, args_len={}",
                            idx, tc.name, tc.arguments.len()
                        );
                        let _ = delta_tx.send(Ok(MessageDelta::ToolCall {
                            id: tc.id,
                            name: tc.name,
                            arguments_delta: tc.arguments,
                        }));
                    }
                }
            }

            flush_buffer(&mut in_think, &mut buffer, true, &delta_tx_clone);

            if let Err(e) = result {
                let _ = delta_tx.send(Err(AgentError::LlmError(format!(
                    "Local LLM inference error: {}",
                    e
                ))));
            }
        });

        let stream = async_stream::stream! {
            while let Some(result) = delta_rx.recv().await {
                yield result;
            }
        };

        Ok(Box::pin(stream))
    }
}
