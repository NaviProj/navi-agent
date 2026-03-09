use tokio::sync::mpsc;

/// Control messages that can be sent to a running agent loop.
///
/// These allow external code to influence the agent's behavior mid-execution,
/// inspired by pi-mono's steering queue pattern.
#[derive(Debug, Clone)]
pub enum AgentControlMessage {
    /// Inject a new user message. The agent will skip remaining tool calls
    /// in the current turn and start a new turn with this message.
    Steer(String),
    /// Graceful cancel: finish the current tool execution, then stop the loop.
    Cancel,
}

/// Sender half of the control channel, held by `NaviAgent`.
pub type ControlSender = mpsc::Sender<AgentControlMessage>;

/// Receiver half of the control channel, passed into `agent_loop`.
pub type ControlReceiver = mpsc::Receiver<AgentControlMessage>;

/// Default buffer size for the control channel.
pub const CONTROL_CHANNEL_BUFFER: usize = 16;
