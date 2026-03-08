use crate::core::event::AgentEvent;
use tokio::sync::mpsc;

pub struct AgentEventStream {
    rx: mpsc::Receiver<AgentEvent>,
}

impl AgentEventStream {
    pub fn new(rx: mpsc::Receiver<AgentEvent>) -> Self {
        Self { rx }
    }

    pub async fn next(&mut self) -> Option<AgentEvent> {
        self.rx.recv().await
    }
}

#[derive(Clone)]
pub struct AgentEventSender {
    tx: mpsc::Sender<AgentEvent>,
}

impl AgentEventSender {
    pub fn new(tx: mpsc::Sender<AgentEvent>) -> Self {
        Self { tx }
    }

    pub async fn send(&self, event: AgentEvent) {
        let _ = self.tx.send(event).await;
    }
}
