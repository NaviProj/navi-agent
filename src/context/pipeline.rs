use crate::core::error::AgentError;
use crate::core::message::NaviMessage;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait ContextTransform: Send + Sync {
    async fn transform(
        &self,
        messages: Vec<NaviMessage>,
        cancel: CancellationToken,
    ) -> Result<Vec<NaviMessage>, AgentError>;
}

#[derive(Default)]
pub struct ContextPipeline {
    transforms: Vec<Box<dyn ContextTransform>>,
}

impl ContextPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_transform(mut self, t: impl ContextTransform + 'static) -> Self {
        self.transforms.push(Box::new(t));
        self
    }

    pub async fn apply(
        &self,
        messages: Vec<NaviMessage>,
        cancel: CancellationToken,
    ) -> Result<Vec<NaviMessage>, AgentError> {
        let mut current = messages;
        for transform in &self.transforms {
            if cancel.is_cancelled() {
                return Err(AgentError::Aborted);
            }
            current = transform.transform(current, cancel.clone()).await?;
        }
        Ok(current)
    }
}
