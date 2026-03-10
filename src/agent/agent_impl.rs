use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent::control::{AgentControlMessage, ControlSender, CONTROL_CHANNEL_BUFFER};
use crate::context::pipeline::ContextPipeline;
use crate::context::store::ContextStore;
use crate::core::error::AgentError;
use crate::runtime::llm_client::LlmClient;
use crate::runtime::loop_impl::{agent_loop, AgentLoopConfig};
use crate::runtime::stream::AgentEventStream;
use crate::tool::middleware::ToolMiddleware;
use crate::tool::registry::ToolRegistry;

/// `NaviAgent` is the primary high-level interface for interacting with the agent framework.
///
/// It wraps the stateless `agent_loop` with persistent state: conversation history,
/// tool registry, context pipeline, and configuration. Each call to `prompt()` or
/// `prompt_with_cancel()` kicks off a new agent turn cycle.
///
/// **Note**: `prompt()` must not be called concurrently. Each call replaces the
/// internal cancel/control handles. If you need concurrent agents, create separate
/// `NaviAgent` instances.
pub struct NaviAgent {
    config: AgentLoopConfig,
    llm_client: Arc<dyn LlmClient>,
    tool_registry: Arc<ToolRegistry>,
    context_pipeline: Arc<ContextPipeline>,
    pub(crate) context_store: Arc<dyn ContextStore>,
    middlewares: Vec<Arc<dyn ToolMiddleware>>,
    cancel: Option<CancellationToken>,
    control_tx: Option<ControlSender>,
    running: Arc<AtomicBool>,
}

impl NaviAgent {
    /// Create a new NaviAgent (prefer using `AgentBuilder`).
    pub(crate) fn new(
        config: AgentLoopConfig,
        llm_client: Arc<dyn LlmClient>,
        tool_registry: Arc<ToolRegistry>,
        context_pipeline: Arc<ContextPipeline>,
        context_store: Arc<dyn ContextStore>,
        middlewares: Vec<Arc<dyn ToolMiddleware>>,
    ) -> Self {
        Self {
            config,
            llm_client,
            tool_registry,
            context_pipeline,
            context_store,
            middlewares,
            cancel: None,
            control_tx: None,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start a new agent cycle with the given user input.
    ///
    /// Returns an `AgentEventStream` for real-time event consumption.
    /// The conversation history is automatically maintained across calls.
    ///
    /// # Errors
    /// Returns `AgentError::AlreadyStreaming` if a previous prompt is still running.
    pub fn prompt(&mut self, input: impl Into<String>) -> Result<AgentEventStream, AgentError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(AgentError::AlreadyStreaming);
        }
        self.running.store(true, Ordering::SeqCst);

        // Cancel any stale previous loop (safety net)
        if let Some(prev_cancel) = self.cancel.take() {
            prev_cancel.cancel();
        }

        let cancel = CancellationToken::new();
        self.cancel = Some(cancel.clone());

        let (control_tx, control_rx) = mpsc::channel(CONTROL_CHANNEL_BUFFER);
        self.control_tx = Some(control_tx);

        let running = self.running.clone();
        let stream = agent_loop(
            self.config.clone(),
            self.llm_client.clone(),
            self.tool_registry.clone(),
            self.context_pipeline.clone(),
            self.context_store.clone(),
            self.middlewares.clone(),
            input.into(),
            cancel,
            Some(control_rx),
        );

        // Spawn a watcher that clears the running flag when the loop task ends.
        // The agent_loop spawns a background task and returns a stream; the stream
        // is backed by an mpsc channel that closes when the task finishes. We rely
        // on abort() / natural completion to clear the flag via this task.
        let running_clone = running;
        let cancel_clone = self.cancel.as_ref().unwrap().clone();
        tokio::spawn(async move {
            cancel_clone.cancelled().await;
            running_clone.store(false, Ordering::SeqCst);
        });

        Ok(stream)
    }

    /// Start a new agent cycle with an external cancellation token.
    ///
    /// # Errors
    /// Returns `AgentError::AlreadyStreaming` if a previous prompt is still running.
    pub fn prompt_with_cancel(
        &mut self,
        input: impl Into<String>,
        cancel: CancellationToken,
    ) -> Result<AgentEventStream, AgentError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(AgentError::AlreadyStreaming);
        }
        self.running.store(true, Ordering::SeqCst);

        // Cancel any stale previous loop
        if let Some(prev_cancel) = self.cancel.take() {
            prev_cancel.cancel();
        }

        self.cancel = Some(cancel.clone());

        let (control_tx, control_rx) = mpsc::channel(CONTROL_CHANNEL_BUFFER);
        self.control_tx = Some(control_tx);

        let running = self.running.clone();
        let stream = agent_loop(
            self.config.clone(),
            self.llm_client.clone(),
            self.tool_registry.clone(),
            self.context_pipeline.clone(),
            self.context_store.clone(),
            self.middlewares.clone(),
            input.into(),
            cancel.clone(),
            Some(control_rx),
        );

        let running_clone = running;
        tokio::spawn(async move {
            cancel.cancelled().await;
            running_clone.store(false, Ordering::SeqCst);
        });

        Ok(stream)
    }

    /// Send a steering message to the running agent.
    ///
    /// The agent will finish executing its current tool, skip any remaining
    /// tool calls in this turn, and start a new turn with this message as
    /// user input.
    ///
    /// Returns `Ok(())` if the message was sent, or `Err` if no agent loop
    /// is currently running.
    pub async fn steer(&self, msg: impl Into<String>) -> Result<(), AgentError> {
        if let Some(tx) = &self.control_tx {
            tx.send(AgentControlMessage::Steer(msg.into()))
                .await
                .map_err(|_| {
                    AgentError::Other(anyhow::anyhow!(
                        "Failed to send steer message: agent loop not running"
                    ))
                })
        } else {
            Err(AgentError::Other(anyhow::anyhow!(
                "No agent loop is currently running"
            )))
        }
    }

    /// Gracefully cancel the running agent. The agent will finish its current
    /// tool execution and then stop, preserving partial results.
    pub async fn graceful_cancel(&self) -> Result<(), AgentError> {
        if let Some(tx) = &self.control_tx {
            tx.send(AgentControlMessage::Cancel)
                .await
                .map_err(|_| {
                    AgentError::Other(anyhow::anyhow!(
                        "Failed to send cancel message: agent loop not running"
                    ))
                })
        } else {
            Err(AgentError::Other(anyhow::anyhow!(
                "No agent loop is currently running"
            )))
        }
    }

    /// Cancel the current running agent loop immediately via CancellationToken.
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
