use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::agent::control::ControlReceiver;
use crate::context::pipeline::ContextPipeline;
use crate::context::store::ContextStore;
use crate::core::error::AgentError;
use crate::core::event::{AgentEvent, MessageDelta};
use crate::core::message::{ContentBlock, LLMMessage, MessageRole, NaviMessage};
use crate::runtime::llm_client::LlmClient;
use crate::runtime::stream::AgentEventStream;
use crate::tool::executor::{ExecutionOutcome, ToolExecutor};
use crate::tool::middleware::ToolMiddleware;
use crate::tool::registry::ToolRegistry;

use futures_util::StreamExt;

/// Configuration for the agent loop.
#[derive(Clone)]
pub struct AgentLoopConfig {
    /// System prompt for the LLM
    pub system_prompt: String,
    /// Maximum number of turns (LLM call → tool execution → LLM call cycles)
    pub max_turns: usize,
    /// Event channel buffer size
    pub event_buffer_size: usize,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_turns: 10,
            event_buffer_size: 256,
        }
    }
}

/// The core agent loop — a stateless function that drives the LLM + Tool cycle.
///
/// This is the "engine" of the agent framework. It takes:
/// - An LLM client for making requests
/// - A tool registry for executing tool calls
/// - A context pipeline for pre-processing messages
/// - An initial set of messages (conversation history)
/// - A user input message
/// - An optional control channel for steering/graceful cancellation
///
/// It returns an `AgentEventStream` that the caller can consume for real-time events.
/// The actual loop runs in a background tokio task.
pub fn agent_loop(
    config: AgentLoopConfig,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<ToolRegistry>,
    context_pipeline: Arc<ContextPipeline>,
    context_store: Arc<dyn ContextStore>,
    middlewares: Vec<Arc<dyn ToolMiddleware>>,
    user_input: String,
    cancel: CancellationToken,
    control_rx: Option<ControlReceiver>,
) -> AgentEventStream {
    let (event_tx, event_rx) = mpsc::channel(config.event_buffer_size);
    let stream = AgentEventStream::new(event_rx);

    tokio::spawn(async move {
        let result = run_loop(
            config,
            llm_client,
            tool_registry,
            context_pipeline,
            context_store,
            middlewares,
            user_input,
            cancel,
            event_tx.clone(),
            control_rx,
        )
        .await;

        if let Err(e) = result {
            let _ = event_tx.send(AgentEvent::Error(e.to_string())).await;
        }
    });

    stream
}

/// Internal loop implementation
async fn run_loop(
    config: AgentLoopConfig,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<ToolRegistry>,
    context_pipeline: Arc<ContextPipeline>,
    context_store: Arc<dyn ContextStore>,
    middlewares: Vec<Arc<dyn ToolMiddleware>>,
    user_input: String,
    cancel: CancellationToken,
    event_tx: mpsc::Sender<AgentEvent>,
    mut control_rx: Option<ControlReceiver>,
) -> Result<(), AgentError> {
    let tool_executor = ToolExecutor::with_middlewares(tool_registry.clone(), middlewares);

    // Load history and build message list
    let mut messages = context_store.load_messages().await?;
    messages.push(NaviMessage::new_user_text(&user_input));

    // Emit AgentStart
    let _ = event_tx.send(AgentEvent::AgentStart).await;

    info!(turns = config.max_turns, "Agent loop started");

    let mut completed_naturally = false;
    for turn in 0..config.max_turns {
        if cancel.is_cancelled() {
            return Err(AgentError::Aborted);
        }

        debug!(turn, "Starting turn");

        // Emit TurnStart
        let _ = event_tx.send(AgentEvent::TurnStart).await;

        // 1. Apply context pipeline (pruning, transformation)
        let processed_messages = context_pipeline
            .apply(messages.clone(), cancel.clone())
            .await?;

        // 2. Get tool definitions
        let tool_defs = tool_registry.definitions();

        // 3. Stream LLM completion (NaviMessage goes directly, no converter needed)
        let mut llm_stream = llm_client
            .stream_completion(&config.system_prompt, &processed_messages, &tool_defs)
            .await?;

        // 4. Accumulate the assistant response
        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<(String, String, serde_json::Value)> = Vec::new();
        let mut thinking_parts: Vec<String> = Vec::new();

        // Build a placeholder assistant message for streaming events
        let streaming_msg = NaviMessage::new_assistant_text("", None);

        // Emit MessageStart
        let _ = event_tx
            .send(AgentEvent::MessageStart {
                message: streaming_msg.clone(),
            })
            .await;

        // Current tool call accumulation state
        let mut current_tc_id: Option<String> = None;
        let mut current_tc_name: Option<String> = None;
        let mut current_tc_args = String::new();

        while let Some(delta_result) = llm_stream.next().await {
            if cancel.is_cancelled() {
                return Err(AgentError::Aborted);
            }

            match delta_result {
                Ok(delta) => {
                    match &delta {
                        MessageDelta::Text(text) => {
                            text_parts.push(text.clone());
                        }
                        MessageDelta::Thinking(text) => {
                            thinking_parts.push(text.clone());
                        }
                        MessageDelta::ToolCall {
                            id,
                            name,
                            arguments_delta,
                        } => {
                            // Handle tool call accumulation
                            // If we see a new id, finalize the previous one
                            if let Some(prev_id) = &current_tc_id {
                                if prev_id != id && !id.is_empty() {
                                    // Finalize previous tool call
                                    match serde_json::from_str::<serde_json::Value>(
                                        &current_tc_args,
                                    ) {
                                        Ok(args_value) => {
                                            tool_calls.push((
                                                prev_id.clone(),
                                                current_tc_name
                                                    .take()
                                                    .unwrap_or_default(),
                                                args_value,
                                            ));
                                        }
                                        Err(e) => {
                                            error!(
                                                args = %current_tc_args,
                                                error = %e,
                                                tool = current_tc_name.as_deref().unwrap_or("unknown"),
                                                "Dropping tool call with malformed JSON arguments"
                                            );
                                            current_tc_name.take();
                                        }
                                    }
                                    current_tc_args.clear();
                                }
                            }

                            if !id.is_empty() {
                                current_tc_id = Some(id.clone());
                            }
                            if !name.is_empty() {
                                current_tc_name = Some(name.clone());
                            }
                            current_tc_args.push_str(arguments_delta);
                        }
                    }

                    // Emit MessageUpdate
                    let _ = event_tx
                        .send(AgentEvent::MessageUpdate {
                            message: streaming_msg.clone(),
                            delta: delta.clone(),
                        })
                        .await;
                }
                Err(e) => {
                    error!(?e, "LLM stream error");
                    let _ = event_tx
                        .send(AgentEvent::Error(format!("LLM stream error: {}", e)))
                        .await;
                    return Err(e);
                }
            }
        }

        // Finalize the last pending tool call if any
        if let Some(tc_id) = current_tc_id.take() {
            match serde_json::from_str::<serde_json::Value>(&current_tc_args) {
                Ok(args_value) => {
                    tool_calls.push((
                        tc_id,
                        current_tc_name.take().unwrap_or_default(),
                        args_value,
                    ));
                }
                Err(e) => {
                    error!(
                        args = %current_tc_args,
                        error = %e,
                        tool = current_tc_name.as_deref().unwrap_or("unknown"),
                        "Dropping tool call with malformed JSON arguments"
                    );
                }
            }
        }

        // Build the final assistant message
        let mut content_blocks = Vec::new();

        let full_thinking = thinking_parts.join("");
        if !full_thinking.is_empty() {
            content_blocks.push(ContentBlock::Thinking(full_thinking));
        }

        let full_text = text_parts.join("");
        if !full_text.is_empty() {
            content_blocks.push(ContentBlock::Text(full_text));
        }

        for (id, name, args) in &tool_calls {
            content_blocks.push(ContentBlock::ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: args.clone(),
            });
        }

        let assistant_msg = NaviMessage::LLM(LLMMessage {
            role: MessageRole::Assistant,
            content: content_blocks,
            timestamp: chrono::Utc::now().timestamp_millis(),
            model: None,
            stop_reason: None,
            error_message: None,
        });

        // Emit MessageEnd
        let _ = event_tx
            .send(AgentEvent::MessageEnd {
                message: assistant_msg.clone(),
            })
            .await;

        // Add assistant message to history
        messages.push(assistant_msg.clone());

        // 5. If there are tool calls, execute them, add results, and continue the loop
        if !tool_calls.is_empty() {
            info!(count = tool_calls.len(), "Executing tool calls");

            let outcome = tool_executor
                .execute_tool_calls(tool_calls, cancel.clone(), &event_tx, &mut control_rx)
                .await?;

            match outcome {
                ExecutionOutcome::Completed(tool_results) => {
                    // Emit TurnEnd
                    let _ = event_tx
                        .send(AgentEvent::TurnEnd {
                            message: assistant_msg,
                            tool_results: tool_results.clone(),
                        })
                        .await;

                    // Add tool results to message history for next turn
                    messages.extend(tool_results);

                    // Continue to next turn — LLM needs to see tool results
                    continue;
                }
                ExecutionOutcome::Steered {
                    results: tool_results,
                    message: steer_msg,
                } => {
                    info!("Agent steered with new message, starting new turn");

                    // Emit TurnEnd with partial results
                    let _ = event_tx
                        .send(AgentEvent::TurnEnd {
                            message: assistant_msg,
                            tool_results: tool_results.clone(),
                        })
                        .await;

                    // Emit Steered event so consumers know the agent was redirected
                    let _ = event_tx
                        .send(AgentEvent::Steered {
                            message: steer_msg.clone(),
                        })
                        .await;

                    // Add partial tool results + steering user message
                    messages.extend(tool_results);
                    messages.push(NaviMessage::new_user_text(&steer_msg));

                    // Continue to next turn with the steering message
                    continue;
                }
                ExecutionOutcome::Cancelled(tool_results) => {
                    info!("Agent gracefully cancelled after tool execution");

                    // Emit TurnEnd with partial results
                    let _ = event_tx
                        .send(AgentEvent::TurnEnd {
                            message: assistant_msg,
                            tool_results: tool_results.clone(),
                        })
                        .await;

                    // Emit GracefullyCancelled event
                    let _ = event_tx
                        .send(AgentEvent::GracefullyCancelled)
                        .await;

                    // Add partial tool results
                    messages.extend(tool_results);

                    // Break out — graceful cancel
                    completed_naturally = true;
                    break;
                }
            }
        }

        // No tool calls — this is the final response
        let _ = event_tx
            .send(AgentEvent::TurnEnd {
                message: assistant_msg,
                tool_results: vec![],
            })
            .await;

        debug!(turn, "Agent loop completed - final response received");
        completed_naturally = true;
        break;
    }

    if !completed_naturally {
        // for-loop exhausted without a natural break — max_turns reached
        error!(
            max_turns = config.max_turns,
            "Agent loop exhausted max_turns without reaching a final response"
        );
        let _ = event_tx
            .send(AgentEvent::Error(format!(
                "Max turns ({}) exhausted without final response",
                config.max_turns
            )))
            .await;
    }

    // Save state before finishing
    context_store.save_messages(&messages).await?;

    // Emit AgentEnd
    let _ = event_tx
        .send(AgentEvent::AgentEnd {
            messages: messages.clone(),
        })
        .await;

    info!("Agent loop finished");
    Ok(())
}
