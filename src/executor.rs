use crate::multiplexer::OutputMultiplexer;
use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{error, info};

#[allow(async_fn_in_trait)]
pub trait AgentExecutor {
    async fn execute(
        &self,
        script: &str,
        interpreter: &str,
        multiplexer: OutputMultiplexer,
    ) -> Result<()>;
}

pub struct LocalExecutor;

impl Default for LocalExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl AgentExecutor for LocalExecutor {
    async fn execute(
        &self,
        script: &str,
        interpreter: &str,
        multiplexer: OutputMultiplexer,
    ) -> Result<()> {
        info!("Starting local execution using {}", interpreter);

        let mut cmd = Command::new(interpreter);
        cmd.arg("-c").arg(script);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().context("Failed to spawn command")?;

        let stdout = child.stdout.take().expect("Child has no stdout");
        let stderr = child.stderr.take().expect("Child has no stderr");

        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        let multiplexer = multiplexer.clone();

        // Notify start
        multiplexer
            .emit_state("running", "Starting execution...")
            .await;

        loop {
            tokio::select! {
                line = stdout_reader.next_line() => {
                    match line {
                        Ok(Some(text)) => {
                            multiplexer.emit_log("stdout", &text).await;
                        }
                        Ok(None) => break, // EOF
                        Err(e) => {
                            error!("Error reading stdout: {}", e);
                            break;
                        }
                    }
                }
                line = stderr_reader.next_line() => {
                     match line {
                        Ok(Some(text)) => {
                            multiplexer.emit_log("stderr", &text).await;
                        }
                        Ok(None) => break, // EOF
                        Err(e) => {
                            error!("Error reading stderr: {}", e);
                            break;
                        }
                    }
                }
                status = child.wait() => {
                    match status {
                        Ok(s) => {
                            info!("Command finished with status: {}", s);
                            if s.success() {
                                multiplexer.emit_state("finished", "Execution successful").await;
                            } else {
                                multiplexer.emit_state("error", &format!("Exited with code: {:?}", s.code())).await;
                            }
                        }
                        Err(e) => {
                            error!("Command failed to wait: {}", e);
                            multiplexer.emit_state("error", &e.to_string()).await;
                        }
                    }
                    break;
                }
            }
        }

        Ok(())
    }
}
