use crate::core::error::AgentError;
use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub details: serde_json::Value,
}

#[async_trait]
pub trait NaviTool: Send + Sync {
    fn name(&self) -> &str;
    fn label(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;

    async fn execute(
        &self,
        tool_call_id: &str,
        params: Value,
        cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError>;
}
