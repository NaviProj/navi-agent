//! Anthropic Messages API serializer and parser.

use crate::core::event::MessageDelta;
use crate::core::message::{ContentBlock, MessageRole, NaviMessage};
use crate::llm::parser::ResponseParser;
use crate::llm::serializer::{MessageSerializer, ToolDef};
use serde_json::{json, Value};

// ─── Serializer ───

pub struct AnthropicSerializer {
    pub max_tokens: u32,
}

impl Default for AnthropicSerializer {
    fn default() -> Self {
        Self { max_tokens: 4096 }
    }
}

impl AnthropicSerializer {
    /// Merge consecutive messages with the same role (Anthropic requirement).
    fn merge_consecutive_roles(messages: Vec<Value>) -> Vec<Value> {
        let mut merged: Vec<Value> = Vec::new();
        for msg in messages {
            if let Some(last) = merged.last_mut() {
                if last["role"] == msg["role"] {
                    if let (Some(last_content), Some(new_content)) =
                        (last["content"].as_array_mut(), msg["content"].as_array())
                    {
                        last_content.extend(new_content.clone());
                        continue;
                    }
                }
            }
            merged.push(msg);
        }
        merged
    }
}

impl MessageSerializer for AnthropicSerializer {
    fn serialize_payload(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        messages: &[NaviMessage],
        tools: &[ToolDef],
        stream: bool,
        thinking: Option<bool>,
    ) -> Value {
        let mut raw_messages = Vec::new();

        for msg in messages {
            match msg {
                NaviMessage::LLM(m) => match m.role {
                    MessageRole::User => {
                        if let Some(text) = m.text() {
                            raw_messages.push(json!({
                                "role": "user",
                                "content": [{"type": "text", "text": text}]
                            }));
                        }
                    }
                    MessageRole::Assistant => {
                        let mut content_blocks = Vec::new();

                        for block in &m.content {
                            match block {
                                ContentBlock::Text(t) => {
                                    if !t.is_empty() {
                                        content_blocks.push(json!({
                                            "type": "text",
                                            "text": t
                                        }));
                                    }
                                }
                                ContentBlock::Thinking(t) => {
                                    content_blocks.push(json!({
                                        "type": "thinking",
                                        "thinking": t
                                    }));
                                }
                                ContentBlock::ToolCall {
                                    id,
                                    name,
                                    arguments,
                                } => {
                                    let parsed_args: Value = if arguments.is_string() {
                                        serde_json::from_str(arguments.as_str().unwrap())
                                            .unwrap_or(json!({}))
                                    } else {
                                        arguments.clone()
                                    };
                                    content_blocks.push(json!({
                                        "type": "tool_use",
                                        "id": id,
                                        "name": name,
                                        "input": parsed_args
                                    }));
                                }
                                _ => {}
                            }
                        }

                        if !content_blocks.is_empty() {
                            raw_messages.push(json!({
                                "role": "assistant",
                                "content": content_blocks
                            }));
                        }
                    }
                    MessageRole::ToolResult => {
                        let mut content_blocks = Vec::new();
                        for block in &m.content {
                            if let ContentBlock::ToolResult {
                                tool_call_id,
                                content,
                                is_error,
                            } = block
                            {
                                content_blocks.push(json!({
                                    "type": "tool_result",
                                    "tool_use_id": tool_call_id,
                                    "content": content,
                                    "is_error": is_error
                                }));
                            }
                        }
                        if !content_blocks.is_empty() {
                            raw_messages.push(json!({
                                "role": "user",
                                "content": content_blocks
                            }));
                        }
                    }
                    _ => {}
                },
                NaviMessage::Custom(_) => {}
            }
        }

        let merged = Self::merge_consecutive_roles(raw_messages);

        let max_tokens = self.max_tokens;

        let mut payload = json!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": merged,
            "stream": stream,
        });

        // Enable extended thinking if requested
        if thinking == Some(true) {
            // When thinking is enabled, budget_tokens should be less than max_tokens.
            // Use 80% of max_tokens as thinking budget, minimum 1024.
            let budget = std::cmp::max(1024, (max_tokens as u64 * 4 / 5) as u32);
            payload["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": budget
            });
        }

        if let Some(sp) = system_prompt {
            payload["system"] = json!(sp);
        }

        if !tools.is_empty() {
            payload["tools"] = json!(tools
                .iter()
                .map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                }))
                .collect::<Vec<_>>());
        }

        payload
    }
}

// ─── Parser ───

pub struct AnthropicParser {
    active_tool_id: String,
    active_tool_name: String,
}

impl AnthropicParser {
    pub fn new() -> Self {
        Self {
            active_tool_id: String::new(),
            active_tool_name: String::new(),
        }
    }
}

impl ResponseParser for AnthropicParser {
    fn parse_sse(&mut self, event_type: &str, data: &str) -> Option<anyhow::Result<MessageDelta>> {
        let parsed: Value = serde_json::from_str(data).ok()?;

        match event_type {
            "content_block_start" => {
                let block = parsed.get("content_block")?;
                if block.get("type")?.as_str()? == "tool_use" {
                    self.active_tool_id = block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    self.active_tool_name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    return Some(Ok(MessageDelta::ToolCall {
                        id: self.active_tool_id.clone(),
                        name: self.active_tool_name.clone(),
                        arguments_delta: String::new(),
                    }));
                }
                None
            }
            "content_block_delta" => {
                let delta = parsed.get("delta")?;

                if let Some(thinking) = delta.get("thinking").and_then(|t| t.as_str()) {
                    return Some(Ok(MessageDelta::Thinking(thinking.to_string())));
                }

                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                    return Some(Ok(MessageDelta::Text(text.to_string())));
                }

                if let Some(partial_json) = delta.get("partial_json").and_then(|j| j.as_str()) {
                    return Some(Ok(MessageDelta::ToolCall {
                        id: self.active_tool_id.clone(),
                        name: self.active_tool_name.clone(),
                        arguments_delta: partial_json.to_string(),
                    }));
                }

                None
            }
            _ => None,
        }
    }
}
