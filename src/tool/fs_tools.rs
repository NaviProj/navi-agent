use crate::core::error::AgentError;
use crate::tool::traits::{NaviTool, ToolResult};
use async_trait::async_trait;
use ignore::{overrides::OverrideBuilder, WalkBuilder};
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};

use std::time::SystemTime;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Deserialize)]
struct ReadParams {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

pub struct ReadTool;

#[async_trait]
impl NaviTool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn label(&self) -> &str {
        "Read File"
    }

    fn description(&self) -> &str {
        "Read file with line numbers (file path, not directory)"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "offset": { "type": "integer" },
                "limit": { "type": "integer" }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let p: ReadParams = serde_json::from_value(params)
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Invalid params: {}", e)))?;

        let content = tokio::fs::read_to_string(&p.path)
            .await
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Read failed: {}", e)))?;

        let mut lines: Vec<&str> = content.split('\n').collect();
        // Remove trailing empty line if string ended with newline split
        if lines.last() == Some(&"") {
            lines.pop();
        }

        let offset = p.offset.unwrap_or(0);
        let limit = p.limit.unwrap_or(lines.len());

        let start = offset.min(lines.len());
        let end = (offset + limit).min(lines.len());
        let selected = &lines[start..end];

        let mut result = String::new();
        for (idx, line) in selected.iter().enumerate() {
            let line_num = offset + idx + 1;
            result.push_str(&format!("{:4}| {}\n", line_num, line));
        }

        Ok(ToolResult {
            content: result,
            is_error: false,
            details: Value::Null,
        })
    }
}

#[derive(Debug, Deserialize)]
struct WriteParams {
    path: String,
    content: String,
}

pub struct WriteTool;

#[async_trait]
impl NaviTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn label(&self) -> &str {
        "Write File"
    }

    fn description(&self) -> &str {
        "Write content to file"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let p: WriteParams = serde_json::from_value(params)
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Invalid params: {}", e)))?;

        tokio::fs::write(&p.path, p.content)
            .await
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Write failed: {}", e)))?;

        Ok(ToolResult {
            content: "ok".to_string(),
            is_error: false,
            details: Value::Null,
        })
    }
}

#[derive(Debug, Deserialize)]
struct EditParams {
    path: String,
    old: String,
    new: String,
    all: Option<bool>,
}

pub struct EditTool;

#[async_trait]
impl NaviTool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn label(&self) -> &str {
        "Edit File"
    }

    fn description(&self) -> &str {
        "Replace old with new in file (old must be unique unless all=true)"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "old": { "type": "string" },
                "new": { "type": "string" },
                "all": { "type": "boolean" }
            },
            "required": ["path", "old", "new"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let p: EditParams = serde_json::from_value(params)
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Invalid params: {}", e)))?;

        let text = tokio::fs::read_to_string(&p.path)
            .await
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Read failed: {}", e)))?;

        if !text.contains(&p.old) {
            return Ok(ToolResult {
                content: "error: old_string not found".to_string(),
                is_error: true,
                details: Value::Null,
            });
        }

        let count = text.matches(&p.old).count();
        let all = p.all.unwrap_or(false);

        if !all && count > 1 {
            return Ok(ToolResult {
                content: format!(
                    "error: old_string appears {} times, must be unique (use all=true)",
                    count
                ),
                is_error: true,
                details: Value::Null,
            });
        }

        let replacement = if all {
            text.replace(&p.old, &p.new)
        } else {
            text.replacen(&p.old, &p.new, 1)
        };

        tokio::fs::write(&p.path, replacement)
            .await
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Write failed: {}", e)))?;

        Ok(ToolResult {
            content: "ok".to_string(),
            is_error: false,
            details: Value::Null,
        })
    }
}

#[derive(Debug, Deserialize)]
struct GlobParams {
    pat: String,
    path: Option<String>,
}

pub struct GlobTool;

#[async_trait]
impl NaviTool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn label(&self) -> &str {
        "Glob Files"
    }

    fn description(&self) -> &str {
        "Find files by pattern, sorted by mtime"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pat": { "type": "string" },
                "path": { "type": "string" }
            },
            "required": ["pat"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let p: GlobParams = serde_json::from_value(params)
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Invalid params: {}", e)))?;

        // File walking and sorting are blocking I/O — run off the async runtime.
        let result = tokio::task::spawn_blocking(move || {
            let base_path = p.path.unwrap_or_else(|| ".".to_string());

            let mut override_builder = OverrideBuilder::new(&base_path);
            let glob_pattern = if p.pat.contains('/') {
                p.pat.clone()
            } else {
                format!("**/{}", p.pat)
            };

            if let Err(e) = override_builder.add(&glob_pattern) {
                return ToolResult {
                    content: format!("Invalid glob pattern: {}", e),
                    is_error: true,
                    details: Value::Null,
                };
            }
            let overrides = override_builder
                .build()
                .unwrap_or_else(|_| ignore::overrides::Override::empty());

            let walker = WalkBuilder::new(&base_path)
                .standard_filters(true)
                .hidden(true)
                .overrides(overrides)
                .build();

            let mut files = Vec::new();
            for result in walker {
                if let Ok(entry) = result {
                    if entry.file_type().map_or(false, |ft| ft.is_file()) {
                        files.push(entry.path().to_path_buf());
                    }
                }
            }

            files.sort_by(|a, b| {
                let time_a = std::fs::metadata(a)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                let time_b = std::fs::metadata(b)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                time_b.cmp(&time_a)
            });

            let file_strs: Vec<String> = files
                .into_iter()
                .map(|f| f.to_string_lossy().into_owned())
                .collect();
            let content = if file_strs.is_empty() {
                "none".to_string()
            } else {
                file_strs.join("\n")
            };

            ToolResult {
                content,
                is_error: false,
                details: Value::Null,
            }
        })
        .await
        .map_err(|e| AgentError::ToolExecutionFailed(format!("Glob task panicked: {}", e)))?;

        Ok(result)
    }
}

#[derive(Debug, Deserialize)]
struct GrepParams {
    pat: String,
    path: Option<String>,
}

pub struct GrepTool;

#[async_trait]
impl NaviTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn label(&self) -> &str {
        "Grep Search"
    }

    fn description(&self) -> &str {
        "Search files for regex pattern"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pat": { "type": "string" },
                "path": { "type": "string" }
            },
            "required": ["pat"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let p: GrepParams = serde_json::from_value(params)
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Invalid params: {}", e)))?;

        // Compile regex before spawning to return errors quickly
        let re = match Regex::new(&p.pat) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!("Regex compile error: {}", e),
                    is_error: true,
                    details: Value::Null,
                });
            }
        };

        // File walking and reading are blocking I/O — run off the async runtime.
        let result = tokio::task::spawn_blocking(move || {
            let base_path = p.path.unwrap_or_else(|| ".".to_string());

            let walker = WalkBuilder::new(&base_path)
                .standard_filters(true)
                .hidden(true)
                .build();

            let mut hits = Vec::new();
            for result in walker {
                if let Ok(entry) = result {
                    if entry.file_type().map_or(false, |ft| ft.is_file()) {
                        let path = entry.path();
                        if let Ok(content) = std::fs::read_to_string(path) {
                            for (line_num, line) in content.lines().enumerate() {
                                if re.is_match(line) {
                                    hits.push(format!(
                                        "{}:{}:{}",
                                        path.display(),
                                        line_num + 1,
                                        line
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            let content = if hits.is_empty() {
                "none".to_string()
            } else {
                hits.iter().take(50).cloned().collect::<Vec<_>>().join("\n")
            };

            ToolResult {
                content,
                is_error: false,
                details: Value::Null,
            }
        })
        .await
        .map_err(|e| AgentError::ToolExecutionFailed(format!("Grep task panicked: {}", e)))?;

        Ok(result)
    }
}

#[derive(Debug, Deserialize)]
struct BashParams {
    cmd: String,
}

pub struct BashTool;

#[async_trait]
impl NaviTool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn label(&self) -> &str {
        "Bash Execute"
    }

    fn description(&self) -> &str {
        "Run shell command"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "cmd": { "type": "string" }
            },
            "required": ["cmd"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let p: BashParams = serde_json::from_value(params)
            .map_err(|e| AgentError::ToolExecutionFailed(format!("Invalid params: {}", e)))?;

        // Executing shell command with 30 seconds timeout
        use std::process::Stdio;
        use tokio::process::Command;

        let child = Command::new("sh")
            .arg("-c")
            .arg(&p.cmd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                AgentError::ToolExecutionFailed(format!("Failed to spawn process: {}", e))
            })?;

        let res =
            tokio::time::timeout(std::time::Duration::from_secs(30), child.wait_with_output())
                .await;

        let content = match res {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let mut out = String::new();
                out.push_str(&stdout);
                if !stderr.is_empty() {
                    out.push_str("\n--- STDERR ---\n");
                    out.push_str(&stderr);
                }
                out.trim().to_string()
            }
            Ok(Err(e)) => format!("Execution failed: {}", e),
            Err(_) => "(timed out after 30s)".to_string(),
        };

        Ok(ToolResult {
            content: if content.is_empty() {
                "(empty)".to_string()
            } else {
                content
            },
            is_error: false,
            details: Value::Null,
        })
    }
}
