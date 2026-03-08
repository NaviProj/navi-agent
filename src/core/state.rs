use super::message::NaviMessage;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct AgentState {
    pub system_prompt: String,
    pub model_id: String,
    pub messages: Vec<NaviMessage>,
    pub is_streaming: bool,
    pub stream_message: Option<NaviMessage>,
    pub pending_tool_calls: HashSet<String>,
    pub error: Option<String>,
}
