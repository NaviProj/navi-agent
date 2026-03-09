use super::traits::NaviTool;
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

    /// Register a tool. If a tool with the same name already exists, it is replaced.
    pub fn register(&self, tool: impl NaviTool + 'static) {
        self.tools
            .write()
            .unwrap()
            .insert(tool.name().to_string(), Arc::new(tool));
    }

    /// Unregister a tool by name. Returns true if the tool was found and removed.
    pub fn unregister(&self, name: &str) -> bool {
        self.tools.write().unwrap().remove(name).is_some()
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn NaviTool>> {
        self.tools.read().unwrap().get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDef> {
        self.tools
            .read()
            .unwrap()
            .values()
            .map(|t| ToolDef {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }
}
