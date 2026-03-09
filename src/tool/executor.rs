use super::middleware::ToolMiddleware;
use super::registry::ToolRegistry;
use crate::agent::control::{AgentControlMessage, ControlReceiver};
use crate::core::error::AgentError;
use crate::core::event::AgentEvent;
use crate::core::message::{ContentBlock, LLMMessage, MessageRole, NaviMessage};
use chrono;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Outcome of tool execution — indicates whether tools completed normally
/// or were interrupted by a control message.
pub enum ExecutionOutcome {
    /// All tool calls completed normally.
    Completed(Vec<NaviMessage>),
    /// A steering message was received after a tool finished.
    /// Contains partial results (tools already executed) and the steering message.
    Steered {
        results: Vec<NaviMessage>,
        message: String,
    },
    /// A graceful cancel was received after a tool finished.
    /// Contains partial results (tools already executed).
    Cancelled(Vec<NaviMessage>),
}

pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
    middlewares: Vec<Arc<dyn ToolMiddleware>>,
}

impl ToolExecutor {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            middlewares: Vec::new(),
        }
    }

    pub fn with_middlewares(
        registry: Arc<ToolRegistry>,
        middlewares: Vec<Arc<dyn ToolMiddleware>>,
    ) -> Self {
        Self {
            registry,
            middlewares,
        }
    }

    /// Execute tool calls with optional control channel for steering/cancellation.
    ///
    /// After each tool execution, the control channel is polled (non-blocking).
    /// If a `Steer` or `Cancel` message is received, remaining tools are skipped.
    pub async fn execute_tool_calls(
        &self,
        tool_calls: Vec<(String, String, Value)>, // (id, name, args)
        cancel: CancellationToken,
        event_tx: &mpsc::Sender<AgentEvent>,
        control_rx: &mut Option<ControlReceiver>,
    ) -> Result<ExecutionOutcome, AgentError> {
        let mut results = Vec::new();
        let total = tool_calls.len();

        for (idx, (id, name, args)) in tool_calls.into_iter().enumerate() {
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

            // Apply before-middlewares (can modify params)
            let mut effective_args = args;
            for mw in &self.middlewares {
                if let Some(modified) = mw.before(&name, &effective_args).await? {
                    effective_args = modified;
                }
            }

            // Execute
            let result = tool.execute(&id, effective_args, cancel.clone()).await;

            match result {
                Ok(res) => {
                    // Apply after-middlewares (reverse order)
                    let mut res = res;
                    for mw in self.middlewares.iter().rev() {
                        res = mw.after(&name, res).await?;
                    }

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

            // After each tool execution, poll the control channel (non-blocking)
            if let Some(rx) = control_rx.as_mut() {
                if let Ok(ctrl_msg) = rx.try_recv() {
                    let remaining = total - idx - 1;
                    if remaining > 0 {
                        tracing::info!(
                            remaining,
                            "Control message received, skipping remaining tool calls"
                        );
                    }
                    match ctrl_msg {
                        AgentControlMessage::Steer(msg) => {
                            return Ok(ExecutionOutcome::Steered {
                                results,
                                message: msg,
                            });
                        }
                        AgentControlMessage::Cancel => {
                            return Ok(ExecutionOutcome::Cancelled(results));
                        }
                    }
                }
            }
        }

        Ok(ExecutionOutcome::Completed(results))
    }
}
