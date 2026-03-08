use crate::core::{ContentBlock, MessageRole, NaviMessage};
use crate::llm::{FunctionCall, Message, ToolCall, UserContent};

pub struct DefaultLlmConverter;

impl DefaultLlmConverter {
    pub fn convert(messages: &[NaviMessage]) -> Vec<Message> {
        messages
            .iter()
            .filter_map(|msg| match msg {
                NaviMessage::LLM(llm_msg) => match llm_msg.role {
                    MessageRole::User => {
                        let texts: Vec<String> = llm_msg
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                ContentBlock::Text(t) => Some(t.clone()),
                                _ => None,
                            })
                            .collect();

                        if texts.is_empty() {
                            None
                        } else {
                            Some(Message::User {
                                content: UserContent::Text(texts.join("\n")),
                            })
                        }
                    }
                    MessageRole::Assistant => {
                        let mut texts = Vec::new();
                        let mut tool_calls = Vec::new();

                        for c in &llm_msg.content {
                            match c {
                                ContentBlock::Text(t) => texts.push(t.clone()),
                                ContentBlock::Thinking(t) => {
                                    texts.push(format!("<think>\n{}\n</think>", t))
                                } // Preserve thinking block in text
                                ContentBlock::ToolCall {
                                    id,
                                    name,
                                    arguments,
                                } => {
                                    tool_calls.push(ToolCall {
                                        id: id.clone(),
                                        r#type: "function".to_string(),
                                        function: FunctionCall {
                                            name: name.clone(),
                                            arguments: if arguments.is_string() {
                                                arguments.as_str().unwrap().to_string()
                                            } else {
                                                arguments.to_string()
                                            },
                                        },
                                    });
                                }
                                _ => {}
                            }
                        }

                        if texts.is_empty() && tool_calls.is_empty() {
                            None
                        } else {
                            Some(Message::Assistant {
                                content: if texts.is_empty() {
                                    None
                                } else {
                                    Some(texts.join("\n\n"))
                                },
                                tool_calls,
                            })
                        }
                    }
                    MessageRole::ToolResult => {
                        let mut tool_id = String::new();
                        let mut tool_content = String::new();

                        for c in &llm_msg.content {
                            if let ContentBlock::ToolResult {
                                tool_call_id,
                                content,
                                ..
                            } = c
                            {
                                tool_id = tool_call_id.clone();
                                tool_content = content.clone();
                            }
                        }

                        if tool_id.is_empty() {
                            None
                        } else {
                            Some(Message::Tool {
                                content: tool_content,
                                tool_call_id: tool_id,
                                name: "".to_string(),
                            })
                        }
                    }
                    _ => None,
                },
                NaviMessage::Custom(_) => None,
            })
            .collect()
    }
}
