use crate::core::error::AgentError;
use crate::tool::traits::ToolResult;
use async_trait::async_trait;
use serde_json::Value;

/// Middleware that wraps tool execution, allowing interception before and after.
///
/// Middlewares are applied in order: `before()` runs first-registered-first,
/// `after()` runs in reverse order (like an onion/stack pattern).
#[async_trait]
pub trait ToolMiddleware: Send + Sync {
    /// Called before tool execution. Return `Ok(Some(params))` to modify params,
    /// `Ok(None)` to pass through unchanged, or `Err` to reject execution.
    async fn before(&self, name: &str, params: &Value) -> Result<Option<Value>, AgentError> {
        let _ = (name, params);
        Ok(None)
    }

    /// Called after tool execution. Can modify the result (e.g., truncate output).
    async fn after(&self, name: &str, result: ToolResult) -> Result<ToolResult, AgentError> {
        let _ = name;
        Ok(result)
    }
}

/// Truncates tool output to a maximum number of characters.
/// Critical for local LLMs with small context windows.
pub struct OutputTruncationMiddleware {
    max_chars: usize,
}

impl OutputTruncationMiddleware {
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

#[async_trait]
impl ToolMiddleware for OutputTruncationMiddleware {
    async fn after(&self, _name: &str, mut result: ToolResult) -> Result<ToolResult, AgentError> {
        if result.content.len() > self.max_chars {
            let truncated = &result.content[..result.content.floor_char_boundary(self.max_chars)];
            result.content = format!(
                "{}\n\n[Output truncated: {} chars total, showing first {}]",
                truncated,
                result.content.len(),
                self.max_chars
            );
        }
        Ok(result)
    }
}
