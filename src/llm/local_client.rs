//! 本地 LLM Client
//!
//! 基于 navi-llm 的 GGUF 模型推理能力，提供与 OpenAI API 兼容的流式对话接口。
//!
//! 由于 llama-cpp-2 的 LlamaContext 不是 Send 的，本模块使用专用线程运行推理，
//! 通过 channel 与 async 世界通信。
use crate::llm::traits::{ChatClient, ChatDelta};
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::Stream;
use navi_llm::{LlmConfig, LlmSessionFactory};
use std::pin::Pin;
use std::sync::mpsc as std_mpsc;
use std::thread;
use tokio::sync::oneshot;

/// 本地 LLM Client
pub struct LocalLMClient {
    /// 发送请求给 worker 线程
    request_tx: std_mpsc::Sender<WorkerRequest>,
    /// worker 线程 handle
    _worker_handle: thread::JoinHandle<()>,
    /// 是否开启思考
    enable_thinking: bool,
}

#[async_trait]
impl ChatClient for LocalLMClient {
    async fn chat_stream(
        &mut self,
        message: String,
        image_bytes: Option<Vec<u8>>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatDelta>> + Send>>> {
        let client = self.clone_inner();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let enable_thinking = client.enable_thinking;
        let tx_clone = tx.clone();

        // Spawn the worker interaction task
        tokio::spawn(async move {
            let state = std::sync::Arc::new(std::sync::Mutex::new((false, String::new())));

            let flush_buffer =
                move |in_think: &mut bool,
                      buffer: &mut String,
                      is_end: bool,
                      tx: &tokio::sync::mpsc::UnboundedSender<Result<ChatDelta>>| {
                    loop {
                        if *in_think {
                            if let Some(pos) = buffer.find("</think>") {
                                let before = &buffer[..pos];
                                if enable_thinking && !before.is_empty() {
                                    let _ = tx.send(Ok(ChatDelta::Thinking(before.to_string())));
                                }
                                *in_think = false;
                                let mut after = buffer[pos + 8..].to_string();
                                if after.starts_with('\n') {
                                    after = after[1..].to_string();
                                }
                                *buffer = after;
                            } else if is_end {
                                if enable_thinking && !buffer.is_empty() {
                                    let _ = tx.send(Ok(ChatDelta::Thinking(buffer.to_string())));
                                }
                                buffer.clear();
                                break;
                            } else {
                                let safe_len = buffer.floor_char_boundary(buffer.len().saturating_sub(8));
                                if safe_len > 0 {
                                    let chunk = buffer[..safe_len].to_string();
                                    if enable_thinking {
                                        let _ = tx.send(Ok(ChatDelta::Thinking(chunk)));
                                    }
                                    buffer.drain(..safe_len);
                                }
                                break;
                            }
                        } else {
                            if let Some(pos) = buffer.find("<think>") {
                                let before = &buffer[..pos];
                                if !before.is_empty() {
                                    let _ = tx.send(Ok(ChatDelta::Text(before.to_string())));
                                }
                                *in_think = true;
                                let mut after = buffer[pos + 7..].to_string();
                                if after.starts_with('\n') {
                                    after = after[1..].to_string();
                                }
                                *buffer = after;
                            } else if is_end {
                                if !buffer.is_empty() {
                                    let _ = tx.send(Ok(ChatDelta::Text(buffer.to_string())));
                                }
                                buffer.clear();
                                break;
                            } else {
                                let safe_len = buffer.floor_char_boundary(buffer.len().saturating_sub(7));
                                if safe_len > 0 {
                                    let chunk = buffer[..safe_len].to_string();
                                    let _ = tx.send(Ok(ChatDelta::Text(chunk)));
                                    buffer.drain(..safe_len);
                                }
                                break;
                            }
                        }
                    }
                };

            let state_clone = state.clone();
            let tx_callback = tx_clone.clone();
            let res = client
                .chat_streaming(message, image_bytes, move |token| {
                    let mut lock = state_clone.lock().unwrap();
                    let (in_think, buffer) = &mut *lock;
                    buffer.push_str(token);
                    flush_buffer(in_think, buffer, false, &tx_callback);
                })
                .await;

            // Final flush
            let mut lock = state.lock().unwrap();
            let (in_think, buffer) = &mut *lock;
            flush_buffer(in_think, buffer, true, &tx_clone);

            if let Err(e) = res {
                let _ = tx.send(Err(anyhow::anyhow!(e)));
            }
            // Channel closes when tx is dropped
        });

        // Convert rx to Stream
        let stream = async_stream::try_stream! {
            while let Some(result) = rx.recv().await {
                yield result?;
            }
        };

        Ok(Box::pin(stream))
    }

    fn add_assistant_message(&mut self, _message: String) -> Result<()> {
        Ok(())
    }

    fn add_message(&mut self, _message: crate::llm::models::Message) -> Result<()> {
        Ok(())
    }

    fn set_tools(&mut self, _tools: Vec<crate::llm::models::ToolDefinition>) {
        tracing::warn!("LocalLlmClient: set_tools ignored");
    }

    fn set_system_prompt(&mut self, _prompt: String) {
        tracing::warn!("LocalLlmClient: set_system_prompt ignored (not supported dynamically yet)");
    }

    fn clear_history(&mut self) {
        let (tx, _) = oneshot::channel();
        let _ = self.request_tx.send(WorkerRequest::Reset { result_tx: tx });
    }

    async fn reset(&mut self) -> Result<()> {
        self.reset_session().await
    }

    fn set_enable_thinking(&mut self, enabled: bool) {
        self.enable_thinking = enabled;
    }
}

/// 本地 LLM 会话配置
#[derive(Debug, Clone)]
pub struct LocalLlmClientConfig {
    /// GGUF 模型文件路径
    pub model_path: String,
    /// 上下文大小
    pub ctx_size: u32,
    /// 最大生成 token 数
    pub max_tokens: u32,
    /// 推理线程数
    pub n_threads: Option<i32>,
    /// System prompt
    pub system_prompt: Option<String>,
    /// 是否开启思考
    pub enable_thinking: bool,
}

/// 发送给 worker 线程的请求
enum WorkerRequest {
    /// 流式聊天请求
    Chat {
        message: String,
        image_bytes: Option<Vec<u8>>,
        token_tx: std_mpsc::Sender<String>,
        result_tx: oneshot::Sender<Result<String>>,
    },
    /// 重置 session
    Reset {
        result_tx: oneshot::Sender<Result<()>>,
    },
}

impl LocalLMClient {
    /// 创建新的本地 LLM Client
    pub fn new(
        config: LocalLlmClientConfig,
        shared_factory: Option<std::sync::Arc<LlmSessionFactory>>,
    ) -> Result<Self> {
        // 创建通信 channel
        let (request_tx, request_rx) = std_mpsc::channel::<WorkerRequest>();

        // 克隆配置给 worker 线程
        let worker_config = config.clone();

        // 启动专用 worker 线程
        let worker_handle = thread::spawn(move || {
            Self::worker_loop(worker_config, shared_factory, request_rx);
        });

        Ok(Self {
            request_tx,
            _worker_handle: worker_handle,
            enable_thinking: config.enable_thinking,
        })
    }

    /// Worker 线程主循环
    fn worker_loop(
        config: LocalLlmClientConfig,
        shared_factory: Option<std::sync::Arc<LlmSessionFactory>>,
        request_rx: std_mpsc::Receiver<WorkerRequest>,
    ) {
        // 如果提供了共享工厂，直接使用它（可能需要覆盖选项）
        let factory = if let Some(factory) = shared_factory {
            factory
        } else {
            // 初始化 new LLM
            let llm_config = {
                let system_prompt = if let Some(sp) = &config.system_prompt {
                    sp.clone()
                } else {
                    let now = chrono::Local::now();
                    let date_str = now.format("%Y-%m-%d %H").to_string();
                    let raw_prompt = "You are a helpful assistant.";
                    raw_prompt.replace("%s", &date_str)
                };

                let mut cfg = LlmConfig::new(&config.model_path)
                    .with_ctx_size(config.ctx_size)
                    .with_max_tokens(config.max_tokens)
                    .with_system_prompt(system_prompt)
                    .with_enable_thinking(config.enable_thinking);

                if let Some(threads) = config.n_threads {
                    cfg = cfg.with_threads(threads);
                }
                cfg
            };

            match LlmSessionFactory::new(llm_config) {
                Ok(f) => std::sync::Arc::new(f),
                Err(e) => {
                    tracing::error!(
                        "[LocalLlmClient] Failed to create LLM session factory: {}",
                        e
                    );
                    return;
                }
            }
        };

        // 创建 session
        let mut session = match factory
            .create_session_with_options(Some(config.ctx_size), Some(config.max_tokens))
        {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("[LocalLlmClient] Failed to create LLM session: {}", e);
                return;
            }
        };

        tracing::info!(
            "[LocalLlmClient] Local LLM worker started: {} (vision: {})",
            factory.model_info(),
            factory.has_vision()
        );

        // 处理请求
        while let Ok(request) = request_rx.recv() {
            match request {
                WorkerRequest::Chat {
                    message,
                    image_bytes,
                    token_tx,
                    result_tx,
                } => {
                    let result = if let (true, Some(ref img)) =
                        (factory.has_vision(), &image_bytes)
                    {
                        tracing::info!(
                            "[LocalLlmClient] Vision inference with {} bytes image",
                            img.len()
                        );
                        factory.complete_with_image_bytes_streaming(&message, img, |token| {
                            let _ = token_tx.send(token.to_string());
                        })
                    } else {
                        session.chat_streaming(&message, |token| {
                            let _ = token_tx.send(token.to_string());
                        })
                    };

                    let _ = result_tx.send(result);
                }
                WorkerRequest::Reset { result_tx } => {
                    session.reset();
                    let _ = result_tx.send(Ok(()));
                }
            }
        }
    }

    /// 流式对话
    pub async fn chat_streaming<F>(
        &self,
        message: String,
        image_bytes: Option<Vec<u8>>,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str) + Send + 'static,
    {
        let (token_tx, token_rx) = std_mpsc::channel::<String>();
        let (result_tx, result_rx) = oneshot::channel();

        // 发送请求给 worker
        self.request_tx
            .send(WorkerRequest::Chat {
                message,
                image_bytes,
                token_tx,
                result_tx,
            })
            .map_err(|_| anyhow::anyhow!("Worker thread closed"))?;

        // 在单独的 blocking 任务中处理 token
        let token_handle = tokio::task::spawn_blocking(move || {
            while let Ok(token) = token_rx.recv() {
                callback(&token);
            }
        });

        // 等待结果
        let result = result_rx.await.context("Worker dropped result channel")?;

        // 等待 token 处理完成
        let _ = token_handle.await;

        result
    }

    /// 重置 Session
    pub async fn reset_session(&self) -> Result<()> {
        let (result_tx, result_rx) = oneshot::channel();

        self.request_tx
            .send(WorkerRequest::Reset { result_tx })
            .map_err(|_| anyhow::anyhow!("Worker thread closed"))?;

        result_rx.await.context("Worker dropped result channel")?
    }

    /// 克隆内部引用
    pub fn clone_inner(&self) -> Self {
        Self {
            request_tx: self.request_tx.clone(),
            _worker_handle: thread::spawn(|| {}),
            enable_thinking: self.enable_thinking,
        }
    }
}
