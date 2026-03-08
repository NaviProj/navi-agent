use serde_json::json;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Emit to TTS (via Cortex) — 关键步骤的语音播报
    Speech(String),
    /// Emit to Client (via DataChannel ServerPush) — 详细执行日志
    ServerPush { kind: String, payload: String },
}

#[derive(Clone)]
pub struct OutputMultiplexer {
    task_id: String,
    event_tx: mpsc::Sender<AgentEvent>,
}

impl OutputMultiplexer {
    pub fn new(task_id: String, event_tx: mpsc::Sender<AgentEvent>) -> Self {
        Self { task_id, event_tx }
    }

    /// 语音播报 — 关键步骤的摘要，通过 TTS 读给用户听
    pub async fn emit_speech(&self, text: &str) {
        let _ = self
            .event_tx
            .send(AgentEvent::Speech(text.to_string()))
            .await;
    }

    /// 推送执行步骤 — 详细的执行过程，通过 DataChannel 文本推送到客户端
    ///
    /// 客户端会收到 `ServerPush { kind: "agent_step", payload: {...} }`
    pub async fn emit_step(&self, step: &str, detail: &str) {
        let payload = json!({
            "task_id": self.task_id,
            "step": step,
            "detail": detail,
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        })
        .to_string();

        let _ = self
            .event_tx
            .send(AgentEvent::ServerPush {
                kind: "agent_step".to_string(),
                payload,
            })
            .await;
    }

    /// 推送原始日志 — 用于脚本执行等场景的 stdout/stderr 输出
    pub async fn emit_log(&self, source: &str, text: &str) {
        let payload = json!({
            "task_id": self.task_id,
            "source": source,
            "text": text,
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        })
        .to_string();

        let _ = self
            .event_tx
            .send(AgentEvent::ServerPush {
                kind: "agent_log".to_string(),
                payload,
            })
            .await;
    }

    /// 推送状态变更 — Agent 任务的生命周期状态
    pub async fn emit_state(&self, state: &str, description: &str) {
        let payload = json!({
            "task_id": self.task_id,
            "state": state,
            "task_description": description
        })
        .to_string();

        let _ = self
            .event_tx
            .send(AgentEvent::ServerPush {
                kind: "agent_state".to_string(),
                payload,
            })
            .await;

        // 关键状态变更也语音播报
        let speech_text = match state {
            "running" => format!("正在{}。", description),
            "finished" => "任务完成。".to_string(),
            "error" => "任务执行出错了。".to_string(),
            _ => String::new(),
        };

        if !speech_text.is_empty() {
            self.emit_speech(&speech_text).await;
        }
    }
}
