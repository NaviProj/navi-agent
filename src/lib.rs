pub mod agent;
pub mod context;
pub mod core;
pub mod llm;
pub mod runtime;
pub mod tool; // Added llm module!

// Existing modules (backward compatibility)
pub mod agents;
pub mod executor;
pub mod multiplexer;

use anyhow::Result;
use executor::{AgentExecutor, LocalExecutor};
pub use multiplexer::{AgentEvent, OutputMultiplexer};
use tokio::sync::mpsc;

pub use agents::schedule::create_schedule_navi_agent;
pub use agents::task::create_task_navi_agent;

// ─── New Architecture Re-exports ───

// Core types
pub use core::event::AgentEvent as NewAgentEvent;
pub use core::{AgentError, AgentState, NaviMessage};

// Runtime
pub use runtime::{agent_loop, AgentEventSender, AgentEventStream, AgentLoopConfig, LlmClient};

// Context
pub use context::{ContextPipeline, ContextTransform, MessageBasedPruner};

// Agent
pub use agent::compat::NaviBotLlmClient;
pub use agent::{AgentBuilder, NaviAgent};

// Tool
pub use tool::{NaviTool, ToolRegistry, ToolResult};

// Schedule NaviTools
pub use agents::schedule_tools::{
    AddEventNaviTool, AddMemoNaviTool, ListEventsNaviTool, ListMemosNaviTool,
};

// ─── Legacy AgentManager ───

pub struct AgentManager {
    executor: LocalExecutor,
}

impl Default for AgentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentManager {
    pub fn new() -> Self {
        Self {
            executor: LocalExecutor::new(),
        }
    }

    pub async fn execute_script(
        &self,
        task_id: String,
        script: String,
        interpreter: String,
        event_tx: mpsc::Sender<AgentEvent>,
    ) -> Result<()> {
        let multiplexer = OutputMultiplexer::new(task_id, event_tx);
        self.executor
            .execute(&script, &interpreter, multiplexer)
            .await
    }
}
