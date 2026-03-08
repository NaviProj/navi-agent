use clap::Parser;
use std::env;
use std::io::{self, Write};
use std::sync::Arc;

use navi_agent::agent::builder::AgentBuilder;
use navi_agent::core::event::{AgentEvent, MessageDelta};
use navi_agent::llm::api_client::{APIChatClient, APIFormat};
use navi_agent::tool::fs_tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The API format to use: 'responses', 'anthropic', 'chat', or 'local'
    #[arg(long, default_value = "responses")]
    api: String,

    /// Base URL
    #[arg(long, default_value = "http://192.168.3.211:8111/v1")]
    baseurl: String,

    /// Log directory path
    #[arg(long, default_value = "logs")]
    log_dir: String,

    #[arg(long, default_value_t = true)]
    local_non_session: bool,

    #[arg(long, default_value_t = 4096)]
    local_ctx_size: u32,

    #[arg(long, default_value_t = 1024)]
    local_max_tokens: u32,

    #[arg(long, default_value_t = true)]
    local_enable_thinking: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 文件日志：写入 log_dir 目录
    let file_appender =
        tracing_appender::rolling::daily(&args.log_dir, "nanocode.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let format = match args.api.as_str() {
        "anthropic" => APIFormat::Anthropic,
        "chat" => APIFormat::OpenAIChat,
        "responses" => APIFormat::OpenAIResponses,
        "local" => APIFormat::OpenAIResponses, // Dummy value for local mode
        _ => {
            eprintln!("Invalid API format specified. Using 'responses'");
            APIFormat::OpenAIResponses
        }
    };

    let proxy_key = env::var("OPENROUTER_API_KEY")
        .or_else(|_| env::var("OPENAI_API_KEY"))
        .unwrap_or_else(|_| "nanocode".to_string());

    let model = env::var("MODEL").unwrap_or_else(|_| "qwen".to_string());

    let client: Arc<dyn navi_agent::runtime::llm_client::LlmClient> = if args.api == "local" {
        #[cfg(feature = "local-llm")]
        {
            use navi_agent::llm::local_llm_client::LocalLlmAgentClient;
            use navi_llm::LlmConfig;

            let model_path = std::path::PathBuf::from(&model);
            let config = LlmConfig::new(&model_path)
                .with_ctx_size(args.local_ctx_size)
                .with_max_tokens(args.local_max_tokens)
                .with_enable_thinking(args.local_enable_thinking);

            Arc::new(LocalLlmAgentClient::new_with_context_mode(
                config,
                args.local_non_session,
            )?)
        }
        #[cfg(not(feature = "local-llm"))]
        {
            anyhow::bail!("Local LLM support requires the 'local-llm' feature. Rebuild with: cargo build -p navi-agent --features local-llm");
        }
    } else {
        Arc::new(APIChatClient::new(
            Some(args.baseurl),
            proxy_key,
            model,
            format,
        ))
    };

    let cwd = env::current_dir()?.display().to_string();
    let system_prompt = format!("Concise coding assistant. cwd: {}", cwd);

    let mut agent = AgentBuilder::new(client)
        .system_prompt(system_prompt)
        .tool(ReadTool)
        .tool(WriteTool)
        .tool(EditTool)
        .tool(GlobTool)
        .tool(GrepTool)
        .tool(BashTool)
        .build();

    let dim = "\x1b[2m";
    let bold = "\x1b[1m";
    let blue = "\x1b[34m";
    let cyan = "\x1b[36m";
    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let reset = "\x1b[0m";

    let term_width = 80;
    let separator = format!("{}{}{}", dim, "─".repeat(term_width.min(80)), reset);

    println!(
        "{}nanocode-rs{} | {} | {}{}\n",
        bold, reset, "RigLlmClient", cwd, reset
    );

    let mut rl = rustyline::DefaultEditor::new()?;
    let prompt = format!("{}{}❯{} ", bold, blue, reset);

    loop {
        println!("{}", separator);
        let readline = rl.readline(&prompt);
        let user_input = match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_str())?;
                line
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        };

        let input = user_input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/q" || input == "exit" {
            break;
        }
        if input == "/c" {
            let _ = agent.clear_history().await;
            println!("{}⏺ Cleared conversation{}", green, reset);
            continue;
        }

        let mut stream = agent.prompt(input);

        // Track accumulated text for printing chunks nicely (if needed)
        let mut is_first_text = true;

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::MessageUpdate { delta, .. } => {
                    match delta {
                        MessageDelta::Text(t) => {
                            if is_first_text {
                                print!("\n{}⏺{} ", cyan, reset);
                                is_first_text = false;
                            }
                            print!("{}", t);
                            io::stdout().flush()?;
                        }
                        MessageDelta::Thinking(t) => {
                            if is_first_text {
                                print!("\n{}⏺{} ", cyan, reset);
                                is_first_text = false;
                            }
                            print!("{}{}{}", dim, t, reset);
                            io::stdout().flush()?;
                        }
                        MessageDelta::ToolCall {
                            name: _name,
                            arguments_delta: _arguments_delta,
                            ..
                        } => {
                            // In real streaming we might get arguments incrementally,
                            // but for CLI simple display we could just wait or print incrementally.
                            // The Python script prints it once the tool is *called*.
                            // With the Agent framework, tools are executed automatically in `run_loop`.
                        }
                    }
                }
                AgentEvent::MessageEnd { message } => {
                    if let navi_agent::core::message::NaviMessage::LLM(llm_msg) = message {
                        // Scan for tool calls to print them since we didn't print them neatly during stream
                        for block in &llm_msg.content {
                            if let navi_agent::core::message::ContentBlock::ToolCall {
                                name,
                                arguments,
                                ..
                            } = block
                            {
                                let arg_preview = serde_json::to_string(arguments)
                                    .unwrap_or_else(|_| "".to_string());
                                let preview = if arg_preview.len() > 50 {
                                    format!("{}...", &arg_preview[..50])
                                } else {
                                    arg_preview
                                };
                                println!(
                                    "\n{}⏺ {}{}({}{}{})",
                                    green, name, reset, dim, preview, reset
                                );
                            }
                        }
                    }
                }
                AgentEvent::TurnEnd { tool_results, .. } => {
                    for res in tool_results {
                        if let navi_agent::core::message::NaviMessage::LLM(llm_msg) = res {
                            if llm_msg.is_tool_result() {
                                for block in llm_msg.content {
                                    if let navi_agent::core::message::ContentBlock::ToolResult {
                                        content,
                                        is_error,
                                        ..
                                    } = block
                                    {
                                        let mut text = content.clone();
                                        if is_error {
                                            text = format!("Error: {}", text);
                                        }
                                        let lines: Vec<&str> = text.lines().collect();

                                        let mut preview = if lines.is_empty() {
                                            "".to_string()
                                        } else {
                                            lines[0].chars().take(60).collect::<String>()
                                        };
                                        if lines.len() > 1 {
                                            preview = format!(
                                                "{} ... +{} lines",
                                                preview,
                                                lines.len() - 1
                                            );
                                        } else if let Some(first) = lines.first() {
                                            if first.len() > 60 {
                                                preview.push_str("...");
                                            }
                                        }

                                        println!("  {}⎿  {}{}", dim, preview, reset);
                                    }
                                }
                            }
                        }
                    }
                    is_first_text = true; // reset for next turn
                }
                AgentEvent::Error(e) => {
                    println!("\n{}⏺ Error: {}{}", red, e, reset);
                }
                _ => {}
            }
        }
        println!(); // newline after loop finishes
    }

    Ok(())
}
