use std::sync::Arc;

use crate::agent::agent_impl::NaviAgent;
use crate::context::pipeline::ContextPipeline;
use crate::context::store::{ContextStore, InMemoryContextStore};
use crate::context::ContextTransform;
use crate::runtime::llm_client::LlmClient;
use crate::runtime::loop_impl::AgentLoopConfig;
use crate::tool::registry::ToolRegistry;
use crate::tool::traits::NaviTool;

/// Builder pattern for constructing a `NaviAgent`.
///
/// # Example
/// ```ignore
/// let agent = AgentBuilder::new(my_llm_client)
///     .system_prompt("You are a helpful assistant.")
///     .max_turns(5)
///     .tool(my_tool)
///     .context_transform(my_pruner)
///     .build();
/// ```
pub struct AgentBuilder {
    llm_client: Arc<dyn LlmClient>,
    config: AgentLoopConfig,
    registry: ToolRegistry,
    pipeline: ContextPipeline,
    context_store: Arc<dyn ContextStore>,
}

impl AgentBuilder {
    /// Create a new builder with the given LLM client.
    pub fn new(llm_client: Arc<dyn LlmClient>) -> Self {
        Self {
            llm_client,
            config: AgentLoopConfig::default(),
            registry: ToolRegistry::new(),
            pipeline: ContextPipeline::new(),
            context_store: Arc::new(InMemoryContextStore::new()),
        }
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = prompt.into();
        self
    }

    /// Set the maximum number of turns.
    pub fn max_turns(mut self, n: usize) -> Self {
        self.config.max_turns = n;
        self
    }

    /// Set the event buffer size.
    pub fn event_buffer_size(mut self, n: usize) -> Self {
        self.config.event_buffer_size = n;
        self
    }

    /// Register a tool.
    pub fn tool(mut self, tool: impl NaviTool + 'static) -> Self {
        self.registry.register(tool);
        self
    }

    /// Add a context transform to the pipeline.
    pub fn context_transform(mut self, transform: impl ContextTransform + 'static) -> Self {
        self.pipeline = self.pipeline.add_transform(transform);
        self
    }

    /// Set a custom context store.
    pub fn context_store(mut self, store: Arc<dyn ContextStore>) -> Self {
        self.context_store = store;
        self
    }

    /// Build the `NaviAgent`.
    pub fn build(self) -> NaviAgent {
        NaviAgent::new(
            self.config,
            self.llm_client,
            Arc::new(self.registry),
            Arc::new(self.pipeline),
            self.context_store,
        )
    }
}
