use async_trait::async_trait;
use chrono::Local;
use rusqlite::{params, Connection};
use serde_json::Value;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

use crate::core::error::AgentError;
use crate::tool::traits::{NaviTool, ToolResult};

// ──────────────────── AddMemo ────────────────────

pub struct AddMemoNaviTool {
    pub db_path: PathBuf,
}

#[async_trait]
impl NaviTool for AddMemoNaviTool {
    fn name(&self) -> &str {
        "add_memo"
    }

    fn label(&self) -> &str {
        "添加备忘"
    }

    fn description(&self) -> &str {
        "Add a new memo to the database."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text content of the memo"
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolExecutionFailed("Missing 'content' field".into()))?;

        let db_path = self.db_path.clone();
        let content = content.to_string();

        // Run DB operation in blocking task
        let result = tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&db_path)
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;
            let now = Local::now().to_rfc3339();
            conn.execute(
                "INSERT INTO memos (content, created_at) VALUES (?1, ?2)",
                params![content, now],
            )
            .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;
            Ok::<String, AgentError>(format!("备忘录已添加: {}", content))
        })
        .await
        .map_err(|e| AgentError::ToolExecutionFailed(format!("Task join error: {}", e)))??;

        Ok(ToolResult {
            content: result,
            is_error: false,
            details: Value::Null,
        })
    }
}

// ──────────────────── AddEvent ────────────────────

pub struct AddEventNaviTool {
    pub db_path: PathBuf,
}

#[async_trait]
impl NaviTool for AddEventNaviTool {
    fn name(&self) -> &str {
        "add_event"
    }

    fn label(&self) -> &str {
        "添加日程"
    }

    fn description(&self) -> &str {
        "Add a new event/schedule to the calendar. start_time should be in ISO8601 format."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the event"
                },
                "start_time": {
                    "type": "string",
                    "description": "The start time in ISO8601 format"
                }
            },
            "required": ["title", "start_time"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let title = params
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolExecutionFailed("Missing 'title' field".into()))?
            .to_string();

        let start_time = params
            .get("start_time")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolExecutionFailed("Missing 'start_time' field".into()))?
            .to_string();

        let db_path = self.db_path.clone();

        let result = tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&db_path)
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;
            let now = Local::now().to_rfc3339();
            conn.execute(
                "INSERT INTO events (title, start_time, created_at) VALUES (?1, ?2, ?3)",
                params![title, start_time, now],
            )
            .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;
            Ok::<String, AgentError>(format!("日程已安排: {} @ {}", title, start_time))
        })
        .await
        .map_err(|e| AgentError::ToolExecutionFailed(format!("Task join error: {}", e)))??;

        Ok(ToolResult {
            content: result,
            is_error: false,
            details: Value::Null,
        })
    }
}

// ──────────────────── ListMemos ────────────────────

pub struct ListMemosNaviTool {
    pub db_path: PathBuf,
}

#[async_trait]
impl NaviTool for ListMemosNaviTool {
    fn name(&self) -> &str {
        "list_memos"
    }

    fn label(&self) -> &str {
        "查看备忘"
    }

    fn description(&self) -> &str {
        "List recent memos."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max number of items to return (default 10)"
                }
            }
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let limit = params.get("limit").and_then(|v| v.as_i64()).unwrap_or(10);

        let db_path = self.db_path.clone();

        let result = tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&db_path)
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;
            let mut stmt = conn
                .prepare("SELECT content FROM memos ORDER BY created_at DESC LIMIT ?1")
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;
            let rows = stmt
                .query_map([limit], |row| row.get::<_, String>(0))
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;

            if rows.is_empty() {
                Ok::<String, AgentError>("暂无备忘录".to_string())
            } else {
                Ok(format!("备忘录列表:\n{}", rows.join("\n")))
            }
        })
        .await
        .map_err(|e| AgentError::ToolExecutionFailed(format!("Task join error: {}", e)))??;

        Ok(ToolResult {
            content: result,
            is_error: false,
            details: Value::Null,
        })
    }
}

// ──────────────────── ListEvents ────────────────────

pub struct ListEventsNaviTool {
    pub db_path: PathBuf,
}

#[async_trait]
impl NaviTool for ListEventsNaviTool {
    fn name(&self) -> &str {
        "list_events"
    }

    fn label(&self) -> &str {
        "查看日程"
    }

    fn description(&self) -> &str {
        "List upcoming events."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max number of items to return (default 10)"
                }
            }
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let limit = params
            .get("limit")
            .and_then(|v| v.as_str())
            .and_then(|v| v.parse::<i64>().ok())
            .or_else(|| params.get("limit").and_then(|v| v.as_i64()))
            .unwrap_or(10);

        let db_path = self.db_path.clone();

        let result = tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&db_path)
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;
            let mut stmt = conn
                .prepare("SELECT title, start_time FROM events ORDER BY start_time ASC LIMIT ?1")
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;
            let rows = stmt
                .query_map([limit], |row| {
                    Ok(format!(
                        "{} @ {}",
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?
                    ))
                })
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| AgentError::ToolExecutionFailed(format!("DB error: {}", e)))?;

            if rows.is_empty() {
                Ok::<String, AgentError>("暂无日程安排".to_string())
            } else {
                Ok(format!("日程列表:\n{}", rows.join("\n")))
            }
        })
        .await
        .map_err(|e| AgentError::ToolExecutionFailed(format!("Task join error: {}", e)))??;

        Ok(ToolResult {
            content: result,
            is_error: false,
            details: Value::Null,
        })
    }
}
