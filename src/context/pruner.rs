use super::pipeline::ContextTransform;
use crate::core::error::AgentError;
use crate::core::message::{ContentBlock, MessageRole, NaviMessage};
use async_trait::async_trait;
use std::collections::HashSet;
use tokio_util::sync::CancellationToken;

pub struct MessageBasedPruner {
    pub max_messages: usize,
}

#[async_trait]
impl ContextTransform for MessageBasedPruner {
    async fn transform(
        &self,
        messages: Vec<NaviMessage>,
        _cancel: CancellationToken,
    ) -> Result<Vec<NaviMessage>, AgentError> {
        // TODO: Implement token-based counting for precise context window management.
        // The pruner currently only cares about message counts, which doesn't
        // account for long tool outputs or system prompts.

        if messages.len() <= self.max_messages {
            return Ok(messages);
        }

        let mut kept = Vec::new();
        let mut missing_tool_calls = HashSet::new();

        // Work backwards from history to always include the most recent context.
        for msg in messages.into_iter().rev() {
            // Track tool call/result parity. APIs (OpenAI/Anthropic) fail if a
            // tool_result is present but its matching tool_call is pruned.
            if let NaviMessage::LLM(m) = &msg {
                if m.role == MessageRole::ToolResult {
                    for block in &m.content {
                        if let ContentBlock::ToolResult { tool_call_id, .. } = block {
                            missing_tool_calls.insert(tool_call_id.clone());
                        }
                    }
                } else if m.role == MessageRole::Assistant {
                    for (id, _, _) in m.tool_calls() {
                        missing_tool_calls.remove(&id);
                    }
                }
            }

            kept.push(msg);

            // Stop if we have enough messages AND all kept tool results have their calls included.
            if kept.len() >= self.max_messages && missing_tool_calls.is_empty() {
                break;
            }
        }

        kept.reverse();
        Ok(kept)
    }
}
