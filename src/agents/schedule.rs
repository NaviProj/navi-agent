use anyhow::{Context, Result};
use chrono::Local;
use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::compat::NaviBotLlmClient;
use crate::agent::{AgentBuilder, NaviAgent};
use crate::agents::schedule_tools::{
    AddEventNaviTool, AddMemoNaviTool, ListEventsNaviTool, ListMemosNaviTool,
};
use crate::context::MessageBasedPruner;

/// 使用新 NaviAgent 框架构建 ScheduleAgent。
///
/// 初始化 SQLite 数据库，创建由 NaviBotLlmClient 驱动的 NaviAgent，
/// 并注册所有日程/备忘工具，系统可直接调用 `agent.prompt(input)` 执行。
pub fn create_schedule_navi_agent(
    db_path: PathBuf,
    api_base: Option<String>,
    api_key: String,
    model: String,
    use_responses_api: bool,
) -> Result<NaviAgent> {
    // Ensure DB tables exist
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

    let llm_client = Arc::new(NaviBotLlmClient::new(
        api_base,
        api_key,
        model,
        use_responses_api,
        false,
    ));

    let system_prompt = format!(
        "你是一个日程和备忘助手。当前时间: {}。",
        Local::now().to_rfc3339()
    );

    let agent = AgentBuilder::new(llm_client)
        .system_prompt(system_prompt)
        .max_turns(5)
        .context_transform(MessageBasedPruner { max_messages: 1024 })
        .tool(AddMemoNaviTool {
            db_path: db_path.clone(),
        })
        .tool(AddEventNaviTool {
            db_path: db_path.clone(),
        })
        .tool(ListMemosNaviTool {
            db_path: db_path.clone(),
        })
        .tool(ListEventsNaviTool { db_path })
        .build();

    Ok(agent)
}
