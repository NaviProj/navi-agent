use anyhow::{Context, Result};
use navi_agent::{
    AddEventNaviTool, AddMemoNaviTool, AgentBuilder, ListEventsNaviTool, ListMemosNaviTool,
    NaviBotLlmClient, NewAgentEvent,
};
use rusqlite::Connection;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let api_key = env::var("OPENAI_API_KEY").context("OPENAI_API_KEY must be set")?;
    let api_base = env::var("OPENAI_API_BASE").ok();
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let instruction = env::args()
        .nth(1)
        .unwrap_or_else(|| "查看我的日程".to_string());

    let db_path = PathBuf::from("schedule.db");

    // Initialize DB
    let conn = Connection::open(&db_path).context("Failed to open schedule DB")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memos (id INTEGER PRIMARY KEY, content TEXT NOT NULL, created_at TEXT NOT NULL)",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, title TEXT NOT NULL, start_time TEXT NOT NULL, created_at TEXT NOT NULL)",
        [],
    )?;
    drop(conn);

    // Create LlmClient adapter
    let llm_client = Arc::new(NaviBotLlmClient::new(
        api_base, api_key, &model, false, false,
    ));

    // Build the agent using the new framework
    let now = chrono::Local::now().to_rfc3339();
    let mut agent = AgentBuilder::new(llm_client)
        .system_prompt(format!(
            "你是一个日程和备忘助手。当前时间: {}。\n\
             用户可能会要求你添加备忘录、添加日程、或查看已有的备忘录和日程。\n\
             请使用提供的工具来完成任务。",
            now
        ))
        .max_turns(5)
        .tool(AddMemoNaviTool {
            db_path: db_path.clone(),
        })
        .tool(AddEventNaviTool {
            db_path: db_path.clone(),
        })
        .tool(ListMemosNaviTool {
            db_path: db_path.clone(),
        })
        .tool(ListEventsNaviTool {
            db_path: db_path.clone(),
        })
        .build();

    println!("📋 Schedule Agent v2 (NaviAgent Framework)");
    println!("📝 指令: {}", instruction);
    println!("─────────────────────────────────");

    // Start the agent loop and consume events
    let mut stream = agent.prompt(&instruction);

    while let Some(event) = stream.next().await {
        match event {
            NewAgentEvent::AgentStart => {
                println!("🚀 Agent 开始执行");
            }
            NewAgentEvent::TurnStart => {
                println!("🔄 新一轮开始");
            }
            NewAgentEvent::MessageUpdate { delta, .. } => match delta {
                navi_agent::core::event::MessageDelta::Text(text) => {
                    print!("{}", text);
                }
                navi_agent::core::event::MessageDelta::Thinking(text) => {
                    print!("💭 {}", text);
                }
                _ => {}
            },
            NewAgentEvent::ToolExecutionStart {
                tool_name, args, ..
            } => {
                println!("\n⚙️  工具调用: {} args={}", tool_name, args);
            }
            NewAgentEvent::ToolExecutionEnd {
                tool_name,
                result,
                is_error,
                ..
            } => {
                if is_error {
                    println!("❌ {} 失败: {}", tool_name, result);
                } else {
                    println!("✅ {} 结果: {}", tool_name, result);
                }
            }
            NewAgentEvent::TurnEnd { .. } => {
                println!("\n🔄 轮次结束");
            }
            NewAgentEvent::AgentEnd { messages } => {
                println!("─────────────────────────────────");
                println!("🏁 Agent 执行完毕 (共 {} 条消息)", messages.len());
            }
            NewAgentEvent::Error(e) => {
                eprintln!("❌ 错误: {}", e);
            }
            _ => {}
        }
    }

    Ok(())
}
