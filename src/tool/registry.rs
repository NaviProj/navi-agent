use super::traits::NaviTool;
use crate::core::error::AgentError;
use crate::llm::serializer::ToolDef;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Registry of tools available to the agent.
///
/// Uses `RwLock` internally to support dynamic tool registration/unregistration
/// while the agent is running. Tool definitions are re-read at the start of each
/// turn, so newly registered tools become available on the next turn.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: Arc<RwLock<HashMap<String, Arc<dyn NaviTool>>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    fn read_tools(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, Arc<dyn NaviTool>>>, AgentError> {
        self.tools.read().map_err(|_| {
            AgentError::Other(anyhow::anyhow!(
                "ToolRegistry RwLock poisoned (a thread panicked while holding the lock)"
            ))
        })
    }

    fn write_tools(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<dyn NaviTool>>>, AgentError>
    {
        self.tools.write().map_err(|_| {
            AgentError::Other(anyhow::anyhow!(
                "ToolRegistry RwLock poisoned (a thread panicked while holding the lock)"
            ))
        })
    }

    /// Register a tool. If a tool with the same name already exists, it is replaced.
    pub fn register(&self, tool: impl NaviTool + 'static) {
        if let Ok(mut guard) = self.write_tools() {
            guard.insert(tool.name().to_string(), Arc::new(tool));
        } else {
            tracing::error!("Failed to register tool: ToolRegistry lock poisoned");
        }
    }

    /// Unregister a tool by name. Returns true if the tool was found and removed.
    pub fn unregister(&self, name: &str) -> bool {
        match self.write_tools() {
            Ok(mut guard) => guard.remove(name).is_some(),
            Err(_) => {
                tracing::error!("Failed to unregister tool: ToolRegistry lock poisoned");
                false
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn NaviTool>> {
        self.read_tools().ok()?.get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDef> {
        match self.read_tools() {
            Ok(guard) => guard
                .values()
                .map(|t| ToolDef {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    parameters: t.parameters_schema(),
                })
                .collect(),
            Err(_) => {
                tracing::error!("Failed to read tool definitions: ToolRegistry lock poisoned");
                Vec::new()
            }
        }
    }
}
