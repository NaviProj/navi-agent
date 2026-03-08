use crate::core::error::AgentError;
use crate::core::message::NaviMessage;
use async_trait::async_trait;
use tokio::sync::RwLock;

#[async_trait]
pub trait ContextStore: Send + Sync {
    async fn load_messages(&self) -> Result<Vec<NaviMessage>, AgentError>;
    async fn save_messages(&self, messages: &[NaviMessage]) -> Result<(), AgentError>;
    async fn clear(&self) -> Result<(), AgentError>;
}

pub struct InMemoryContextStore {
    messages: RwLock<Vec<NaviMessage>>,
}

impl Default for InMemoryContextStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryContextStore {
    pub fn new() -> Self {
        Self {
            messages: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl ContextStore for InMemoryContextStore {
    async fn load_messages(&self) -> Result<Vec<NaviMessage>, AgentError> {
        let guard = self.messages.read().await;
        Ok(guard.clone())
    }

    async fn save_messages(&self, messages: &[NaviMessage]) -> Result<(), AgentError> {
        let mut guard = self.messages.write().await;
        *guard = messages.to_vec();
        Ok(())
    }

    async fn clear(&self) -> Result<(), AgentError> {
        let mut guard = self.messages.write().await;
        guard.clear();
        Ok(())
    }
}
