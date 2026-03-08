pub mod api_client;
#[cfg(feature = "local-llm")]
pub mod local_client;
#[cfg(feature = "local-llm")]
pub mod local_llm_client;
pub mod models;
pub mod stateful_client;
pub mod traits;

// New modular serializer/parser architecture
pub mod anthropic;
pub mod openai_chat;
pub mod openai_responses;
pub mod parser;
pub mod serializer;

pub use api_client::{APIChatClient, APIFormat};
#[cfg(feature = "local-llm")]
pub use local_client::{LocalLMClient, LocalLlmClientConfig};
#[cfg(feature = "local-llm")]
pub use local_llm_client::LocalLlmAgentClient;
pub use models::*;
pub use serializer::ToolDef;
pub use stateful_client::StatefulChatClient;
pub use traits::{ChatClient, ChatDelta};
