//! OpenAI Responses API serializer and parser.

use crate::core::event::MessageDelta;
use crate::core::message::{ContentBlock, MessageRole, NaviMessage};
use crate::llm::parser::ResponseParser;
use crate::llm::serializer::{MessageSerializer, ToolDef};
use serde_json::{json, Value};

// ─── Serializer ───

pub struct OpenAIResponsesSerializer;

impl MessageSerializer for OpenAIResponsesSerializer {
    fn serialize_payload(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        messages: &[NaviMessage],
        tools: &[ToolDef],
        stream: bool,
        _thinking: Option<bool>,
    ) -> Value {
        let mut items = Vec::new();

        if let Some(sp) = system_prompt {
            items.push(json!({
                "type": "message",
                "role": "system",
                "content": sp
            }));
        }

        for msg in messages {
            match msg {
                NaviMessage::LLM(m) => match m.role {
                    MessageRole::User => {
                        if let Some(text) = m.text() {
                            items.push(json!({
                                "type": "message",
                                "role": "user",
                                "content": text
                            }));
                        }
                    }
                    MessageRole::System => {
                        if let Some(text) = m.text() {
                            items.push(json!({
                                "type": "message",
                                "role": "system",
                                "content": text
                            }));
                        }
                    }
                    MessageRole::Assistant => {
                        let mut text_parts = Vec::new();

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
                                    items.push(json!({
                                        "type": "function_call",
                                        "call_id": id,
                                        "name": name,
                                        "arguments": args_str
                                    }));
                                }
                                _ => {}
                            }
                        }

                        if !text_parts.is_empty() {
                            items.push(json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": text_parts.join("\n\n")
                                    }
                                ]
                            }));
                        }
                    }
                    MessageRole::ToolResult => {
                        for block in &m.content {
                            if let ContentBlock::ToolResult {
                                tool_call_id,
                                content,
                                is_error,
                            } = block
                            {
                                let final_output = if *is_error {
                                    format!("[ERROR] {}", content)
                                } else {
                                    content.clone()
                                };
                                items.push(json!({
                                    "type": "function_call_output",
                                    "call_id": tool_call_id,
                                    "output": final_output
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
            "input": items,
            "stream": stream,
        });

        if !tools.is_empty() {
            payload["tools"] = json!(tools
                .iter()
                .map(|t| json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }))
                .collect::<Vec<_>>());
        }

        payload
    }
}

// ─── Parser ───

pub struct OpenAIResponsesParser {
    /// Maps item_id → call_id for function call items.
    /// The Responses API uses `item_id` in argument delta events, but the actual
    /// call_id (used for matching tool results) is set in `output_item.added`.
    item_to_call_id: std::collections::HashMap<String, String>,
}

impl OpenAIResponsesParser {
    pub fn new() -> Self {
        Self {
            item_to_call_id: std::collections::HashMap::new(),
        }
    }
}

impl ResponseParser for OpenAIResponsesParser {
    fn parse_sse(&mut self, _event_type: &str, data: &str) -> Option<anyhow::Result<MessageDelta>> {
        if data == "[DONE]" {
            return None;
        }

        let parsed: Value = serde_json::from_str(data).ok()?;

        let type_str = parsed.get("type")?.as_str()?;

        match type_str {
            "response.output_text.delta" => {
                let delta = parsed.get("delta")?.as_str()?;
                Some(Ok(MessageDelta::Text(delta.to_string())))
            }
            "response.reasoning_text.delta" => {
                let delta = parsed.get("delta")?.as_str()?;
                Some(Ok(MessageDelta::Thinking(delta.to_string())))
            }
            "response.output_item.added" => {
                let item = parsed.get("item")?;
                if item.get("type")?.as_str()? == "function_call" {
                    let name = item
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let call_id = item
                        .get("call_id")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    let item_id = item
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Store the mapping from item_id to call_id
                    if !item_id.is_empty() && !call_id.is_empty() {
                        self.item_to_call_id.insert(item_id, call_id.clone());
                    }

                    Some(Ok(MessageDelta::ToolCall {
                        id: call_id,
                        name,
                        arguments_delta: String::new(),
                    }))
                } else {
                    None
                }
            }
            "response.function_call_arguments.delta" => {
                let item_id = parsed
                    .get("item_id")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();
                // Resolve item_id → call_id; fall back to item_id if not found
                let call_id = self
                    .item_to_call_id
                    .get(&item_id)
                    .cloned()
                    .unwrap_or(item_id);
                let delta = parsed
                    .get("delta")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(Ok(MessageDelta::ToolCall {
                    id: call_id,
                    name: String::new(),
                    arguments_delta: delta,
                }))
            }
            _ => None,
        }
    }
}
