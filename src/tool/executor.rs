use super::registry::ToolRegistry;
use crate::core::error::AgentError;
use crate::core::event::AgentEvent;
use crate::core::message::{ContentBlock, LLMMessage, MessageRole, NaviMessage};
use chrono;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
}

impl ToolExecutor {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }

    pub async fn execute_tool_calls(
        &self,
        tool_calls: Vec<(String, String, Value)>, // (id, name, args)
        cancel: CancellationToken,
        event_tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<Vec<NaviMessage>, AgentError> {
        let mut results = Vec::new();

        for (id, name, args) in tool_calls {
            if cancel.is_cancelled() {
                return Err(AgentError::Aborted);
            }

            let tool = self
                .registry
                .get(&name)
                .ok_or_else(|| AgentError::ToolNotFound(name.clone()))?;

            // Emit Start Event
            let _ = event_tx
                .send(AgentEvent::ToolExecutionStart {
                    tool_call_id: id.clone(),
                    tool_name: name.clone(),
                    args: args.clone(),
                })
                .await;

            // Execute
            let result = tool.execute(&id, args, cancel.clone()).await;

            match result {
                Ok(res) => {
                    // Emit End Event
                    let _ = event_tx
                        .send(AgentEvent::ToolExecutionEnd {
                            tool_call_id: id.clone(),
                            tool_name: name.clone(),
                            result: res.content.clone(),
                            is_error: res.is_error,
                        })
                        .await;

                    // Create ToolResult message
                    results.push(NaviMessage::LLM(LLMMessage {
                        role: MessageRole::ToolResult,
                        content: vec![ContentBlock::ToolResult {
                            tool_call_id: id,
                            content: res.content,
                            is_error: res.is_error,
                        }],
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        model: None,
                        stop_reason: None,
                        error_message: None,
                    }));
                }
                Err(e) => {
                    // Emit End Event with error
                    let _ = event_tx
                        .send(AgentEvent::ToolExecutionEnd {
                            tool_call_id: id.clone(),
                            tool_name: name.clone(),
                            result: e.to_string(),
                            is_error: true,
                        })
                        .await;

                    // Create Error ToolResult message
                    results.push(NaviMessage::LLM(LLMMessage {
                        role: MessageRole::ToolResult,
                        content: vec![ContentBlock::ToolResult {
                            tool_call_id: id,
                            content: e.to_string(),
                            is_error: true,
                        }],
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        model: None,
                        stop_reason: None,
                        error_message: None,
                    }));
                }
            }
        }

        Ok(results)
    }
}
