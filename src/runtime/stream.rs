use crate::core::event::AgentEvent;
use std::pin::Pin;
use std::task::{Context, Poll};
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

impl futures_util::Stream for AgentEventStream {
    type Item = AgentEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
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
