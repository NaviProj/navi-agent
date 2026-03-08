use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::context::pipeline::ContextPipeline;
use crate::context::store::ContextStore;
use crate::core::error::AgentError;
use crate::runtime::llm_client::LlmClient;
use crate::runtime::loop_impl::{agent_loop, AgentLoopConfig};
use crate::runtime::stream::AgentEventStream;
use crate::tool::registry::ToolRegistry;

/// `NaviAgent` is the primary high-level interface for interacting with the agent framework.
///
/// It wraps the stateless `agent_loop` with persistent state: conversation history,
/// tool registry, context pipeline, and configuration. Each call to `prompt()` or
/// `prompt_with_cancel()` kicks off a new agent turn cycle.
pub struct NaviAgent {
    config: AgentLoopConfig,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<ToolRegistry>,
    context_pipeline: Arc<ContextPipeline>,
    pub(crate) context_store: Arc<dyn ContextStore>,
    cancel: Option<CancellationToken>,
}

impl NaviAgent {
    /// Create a new NaviAgent (prefer using `AgentBuilder`).
    pub(crate) fn new(
        config: AgentLoopConfig,
        llm_client: Arc<dyn LlmClient>,
        tool_registry: Arc<ToolRegistry>,
        context_pipeline: Arc<ContextPipeline>,
        context_store: Arc<dyn ContextStore>,
    ) -> Self {
        Self {
            config,
            llm_client,
            tool_registry,
            context_pipeline,
            context_store,
            cancel: None,
        }
    }

    /// Start a new agent cycle with the given user input.
    ///
    /// Returns an `AgentEventStream` for real-time event consumption.
    /// The conversation history is automatically maintained across calls.
    pub fn prompt(&mut self, input: impl Into<String>) -> AgentEventStream {
        let cancel = CancellationToken::new();
        self.cancel = Some(cancel.clone());

        agent_loop(
            self.config.clone(),
            self.llm_client.clone(),
            self.tool_registry.clone(),
            self.context_pipeline.clone(),
            self.context_store.clone(),
            input.into(),
            cancel,
        )
    }

    /// Start a new agent cycle with an external cancellation token.
    pub fn prompt_with_cancel(
        &mut self,
        input: impl Into<String>,
        cancel: CancellationToken,
    ) -> AgentEventStream {
        self.cancel = Some(cancel.clone());

        agent_loop(
            self.config.clone(),
            self.llm_client.clone(),
            self.tool_registry.clone(),
            self.context_pipeline.clone(),
            self.context_store.clone(),
            input.into(),
            cancel,
        )
    }

    /// Cancel the current running agent loop.
    pub fn abort(&self) {
        if let Some(cancel) = &self.cancel {
            cancel.cancel();
        }
    }

    pub async fn clear_history(&self) -> Result<(), AgentError> {
        self.context_store.clear().await
    }

    /// Update the system prompt.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.config.system_prompt = prompt.into();
    }

    /// Get a reference to the tool registry.
    pub fn tool_registry(&self) -> &ToolRegistry {
        &self.tool_registry
    }
}
