//! OpenAI Chat Completions API serializer and parser.

use crate::core::event::MessageDelta;
use crate::core::message::{ContentBlock, MessageRole, NaviMessage};
use crate::llm::parser::ResponseParser;
use crate::llm::serializer::{MessageSerializer, ToolDef};
use serde_json::{json, Value};

// ─── Serializer ───

pub struct OpenAIChatSerializer;

impl MessageSerializer for OpenAIChatSerializer {
    fn serialize_payload(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        messages: &[NaviMessage],
        tools: &[ToolDef],
        stream: bool,
        thinking: Option<bool>,
    ) -> Value {
        let mut msgs = Vec::new();

        if let Some(sp) = system_prompt {
            msgs.push(json!({"role": "system", "content": sp}));
        }

        for msg in messages {
            match msg {
                NaviMessage::LLM(m) => match m.role {
                    MessageRole::User => {
                        if let Some(text) = m.text() {
                            msgs.push(json!({"role": "user", "content": text}));
                        }
                    }
                    MessageRole::System => {
                        if let Some(text) = m.text() {
                            msgs.push(json!({"role": "system", "content": text}));
                        }
                    }
                    MessageRole::Assistant => {
                        let mut text_parts = Vec::new();
                        let mut tool_calls = Vec::new();

                        for block in &m.content {
                            match block {
                                ContentBlock::Text(t) => text_parts.push(t.clone()),
                                ContentBlock::Thinking(t) => {
                                    text_parts.push(format!("<think>\n{}\n</think>", t));
                                }
                                ContentBlock::ToolCall {
                                    id,
                                    name,
                                    arguments,
                                } => {
                                    let args_str = if arguments.is_string() {
                                        arguments.as_str().unwrap().to_string()
                                    } else {
                                        arguments.to_string()
                                    };
                                    tool_calls.push(json!({
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": args_str
                                        }
                                    }));
                                }
                                _ => {}
                            }
                        }

                        let content = if text_parts.is_empty() {
                            None
                        } else {
                            Some(text_parts.join("\n\n"))
                        };

                        let mut obj = json!({"role": "assistant"});
                        if let Some(c) = content {
                            obj["content"] = json!(c);
                        }
                        if !tool_calls.is_empty() {
                            obj["tool_calls"] = json!(tool_calls);
                        }
                        msgs.push(obj);
                    }
                    MessageRole::ToolResult => {
                        for block in &m.content {
                            if let ContentBlock::ToolResult {
                                tool_call_id,
                                content,
                                is_error,
                            } = block
                            {
                                let final_content = if *is_error {
                                    format!("[ERROR] {}", content)
                                } else {
                                    content.clone()
                                };
                                msgs.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": final_content
                                }));
                            }
                        }
                    }
                },
                NaviMessage::Custom(_) => {}
            }
        }

        let mut payload = json!({
            "model": model,
            "messages": msgs,
            "stream": stream,
        });

        // Add thinking parameter for Kimi k2.5
        if model.contains("kimi-k2.5") {
            let type_str = if thinking.unwrap_or(true) {
                "enabled"
            } else {
                "disabled"
            };
            payload["thinking"] = json!({
                "type": type_str
            });
        }

        if !tools.is_empty() {
            payload["tools"] = json!(tools
                .iter()
                .map(|t| json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                }))
                .collect::<Vec<_>>());
        }

        payload
    }
}

// ─── Parser ───

pub struct OpenAIChatParser;

impl OpenAIChatParser {
    pub fn new() -> Self {
        Self
    }
}

impl ResponseParser for OpenAIChatParser {
    fn parse_sse(&mut self, _event_type: &str, data: &str) -> Option<anyhow::Result<MessageDelta>> {
        if data == "[DONE]" {
            return None;
        }

        let parsed: Value = serde_json::from_str(data).ok()?;

        let choices = parsed.get("choices")?.as_array()?;
        let first = choices.first()?;
        let delta = first.get("delta")?;

        // Reasoning content (e.g., DeepSeek, Kimi)
        if let Some(reasoning) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
            if !reasoning.is_empty() {
                return Some(Ok(MessageDelta::Thinking(reasoning.to_string())));
            }
        }

        // Text content
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                return Some(Ok(MessageDelta::Text(content.to_string())));
            }
        }

        // Tool calls
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                if let Some(function) = tc.get("function") {
                    let name = function
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = function
                        .get("arguments")
                        .and_then(|a| a.as_str())
                        .unwrap_or("")
                        .to_string();
                    let id = tc
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();

                    return Some(Ok(MessageDelta::ToolCall {
                        id,
                        name,
                        arguments_delta: args,
                    }));
                }
            }
        }

        None
    }
}
