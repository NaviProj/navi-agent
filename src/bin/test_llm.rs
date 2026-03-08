use clap::Parser;
use futures_util::StreamExt;
use navi_agent::core::event::MessageDelta;
use navi_agent::llm::{APIChatClient, ToolDef};
use navi_agent::runtime::llm_client::LlmClient;
use serde_json::json;
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// OpenAI API Key
    #[arg(long, env = "OPENAI_API_KEY")]
    api_key: String,

    /// OpenAI API Base URL
    #[arg(
        long,
        env = "OPENAI_API_BASE",
        default_value = "https://api.moonshot.cn/v1"
    )]
    api_base: Option<String>,

    /// Model name
    #[arg(long, short, default_value = "kimi-k2.5")]
    model: String,

    /// Prompt to send
    #[arg(short, long, default_value = "Hello!")]
    prompt: String,

    /// System prompt
    #[arg(long)]
    system: Option<String>,

    /// Enable a dummy tool for testing
    #[arg(long)]
    enable_tools: bool,

    /// Use OpenAI Responses API (default: false, uses Completions API)
    #[arg(long)]
    use_responses_api: bool,

    /// Enable thinking parameter (e.g. for kimi k2.5 models)
    #[arg(long)]
    enable_thinking: bool,

    /// Optional image path to read and send with the prompt
    #[arg(short, long)]
    image: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Initializing ApiChatClient...");
    println!("Model: {}", args.model);
    if let Some(base) = &args.api_base {
        println!("API Base: {}", base);
    }
    println!("Use Responses API: {}", args.use_responses_api);

    let client = APIChatClient::detect_with_thinking(
        args.api_base,
        args.api_key,
        args.model,
        args.use_responses_api,
        if args.enable_thinking {
            Some(true)
        } else {
            None
        },
    );

    let mut tools = Vec::new();
    if args.enable_tools {
        println!("Enabling dummy 'get_weather' tool...");
        tools.push(ToolDef {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }),
        });
    }

    let system_prompt = args.system.as_deref().unwrap_or("");
    let user_msg = navi_agent::core::NaviMessage::new_user_text(&args.prompt);

    println!("\nSending prompt: \"{}\"", args.prompt);
    println!("--------------------------------------------------\n");

    let mut stream = client
        .stream_completion(system_prompt, &[user_msg], &tools)
        .await?;

    while let Some(result) = stream.next().await {
        match result {
            Ok(delta) => match delta {
                MessageDelta::Text(text) => {
                    print!("{}", text);
                    io::stdout().flush()?;
                }
                MessageDelta::Thinking(text) => {
                    print!("\x1b[90m{}\x1b[0m", text); // print in gray for thinking
                    io::stdout().flush()?;
                }
                MessageDelta::ToolCall {
                    name,
                    arguments_delta,
                    id,
                } => {
                    println!("\n\n[Tool Call Detected]");
                    println!("  ID: {}", id);
                    println!("  Name: {}", name);
                    println!("  Arguments: {}", arguments_delta);
                }
            },
            Err(e) => {
                eprintln!("\n\nError receiving stream: {}", e);
                break;
            }
        }
    }

    println!("\n\n--------------------------------------------------");
    println!("Stream finished.");

    Ok(())
}
