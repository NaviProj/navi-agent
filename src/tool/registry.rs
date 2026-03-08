use super::traits::NaviTool;
use crate::llm::serializer::ToolDef;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn NaviTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: impl NaviTool + 'static) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn NaviTool>> {
        self.tools.get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDef> {
        self.tools
            .values()
            .map(|t| ToolDef {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }
}
