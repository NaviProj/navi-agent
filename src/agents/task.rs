use anyhow::Result;
use std::sync::Arc;

use crate::agent::builder::AgentBuilder;
use crate::agent::compat::NaviBotLlmClient;
use crate::agent::NaviAgent;
use crate::tool::fs_tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool};

pub fn create_task_navi_agent(
    api_base: Option<String>,
    api_key: String,
    model_name: String,
    use_responses_api: bool,
) -> Result<NaviAgent> {
    let client = Arc::new(NaviBotLlmClient::new(
        api_base,
        api_key,
        model_name,
        use_responses_api,
        false,
    ));

    let system_prompt = "You are a helpful and concise coding assistant. You have access to the file system and shell. Do NOT write long explanations. Just execute the steps.".to_string();

    let agent = AgentBuilder::new(client)
        .system_prompt(system_prompt)
        .tool(ReadTool)
        .tool(WriteTool)
        .tool(EditTool)
        .tool(GlobTool)
        .tool(GrepTool)
        .tool(BashTool)
        .build();

    Ok(agent)
}
