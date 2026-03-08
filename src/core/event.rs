use super::message::NaviMessage;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    // Agent Lifecycle
    AgentStart,
    AgentEnd {
        messages: Vec<NaviMessage>,
    },

    // Turn Lifecycle
    TurnStart,
    TurnEnd {
        message: NaviMessage,
        tool_results: Vec<NaviMessage>,
    },

    // Message Lifecycle
    MessageStart {
        message: NaviMessage,
    },
    MessageUpdate {
        message: NaviMessage,
        delta: MessageDelta,
    },
    MessageEnd {
        message: NaviMessage,
    },

    // Tool Execution Lifecycle
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        partial_result: String,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: String,
        is_error: bool,
    },

    // Error
    Error(String),
}

#[derive(Debug, Clone)]
pub enum MessageDelta {
    Text(String),
    Thinking(String),
    ToolCall {
        id: String,
        name: String,
        arguments_delta: String,
    },
}
