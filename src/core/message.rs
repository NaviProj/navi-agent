use chrono;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    ToolResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum NaviMessage {
    LLM(LLMMessage),
    Custom(CustomMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMMessage {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
    pub timestamp: i64,
    // Optional metadata for Assistant messages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ContentBlock {
    Text(String),
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
        is_error: bool,
    },
    Thinking(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMessage {
    pub kind: String,
    pub payload: serde_json::Value,
    pub timestamp: i64,
}

impl NaviMessage {
    pub fn role(&self) -> String {
        match self {
            NaviMessage::LLM(msg) => format!("{:?}", msg.role),
            NaviMessage::Custom(msg) => msg.kind.clone(),
        }
    }

    pub fn new_user_text(text: impl Into<String>) -> Self {
        Self::LLM(LLMMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text(text.into())],
            timestamp: chrono::Utc::now().timestamp_millis(),
            model: None,
            stop_reason: None,
            error_message: None,
        })
    }

    pub fn new_assistant_text(text: impl Into<String>, model: Option<String>) -> Self {
        Self::LLM(LLMMessage {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text(text.into())],
            timestamp: chrono::Utc::now().timestamp_millis(),
            model,
            stop_reason: None,
            error_message: None,
        })
    }

    pub fn new_custom(kind: impl Into<String>, payload: serde_json::Value) -> Self {
        Self::Custom(CustomMessage {
            kind: kind.into(),
            payload,
            timestamp: chrono::Utc::now().timestamp_millis(),
        })
    }

    pub fn new_tool_call(id: String, name: String, arguments: serde_json::Value) -> Self {
        Self::LLM(LLMMessage {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::ToolCall {
                id,
                name,
                arguments,
            }],
            timestamp: chrono::Utc::now().timestamp_millis(),
            model: None,
            stop_reason: None,
            error_message: None,
        })
    }

    pub fn new_tool_result(tool_call_id: String, content: String, is_error: bool) -> Self {
        Self::LLM(LLMMessage {
            role: MessageRole::ToolResult,
            content: vec![ContentBlock::ToolResult {
                tool_call_id,
                content,
                is_error,
            }],
            timestamp: chrono::Utc::now().timestamp_millis(),
            model: None,
            stop_reason: None,
            error_message: None,
        })
    }
}

impl LLMMessage {
    pub fn text(&self) -> Option<String> {
        let texts: Vec<_> = self
            .content
            .iter()
            .filter_map(|c| {
                if let ContentBlock::Text(t) = c {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();

        if texts.is_empty() {
            None
        } else {
            Some(texts.join("\n"))
        }
    }

    pub fn tool_calls(&self) -> Vec<(String, String, serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|c| {
                if let ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } = c
                {
                    Some((id.clone(), name.clone(), arguments.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn is_tool_call(&self) -> bool {
        self.content
            .iter()
            .any(|c| matches!(c, ContentBlock::ToolCall { .. }))
    }

    pub fn is_tool_result(&self) -> bool {
        self.role == MessageRole::ToolResult
    }
}
