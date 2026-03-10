//! Shared think-tag parser for `<think>...</think>` blocks in LLM output.
//!
//! Used by both `local_llm_client` (new architecture, `MessageDelta`) and
//! `local_client` (legacy `ChatClient`, `ChatDelta`) to avoid duplicating
//! the same streaming state machine.

/// Events emitted by the think-tag parser.
#[derive(Debug, Clone)]
pub enum ThinkTagEvent {
    /// Regular text content outside `<think>` blocks.
    Text(String),
    /// Thinking content inside `<think>` blocks.
    Thinking(String),
}

/// Streaming parser that separates `<think>...</think>` blocks from regular text.
///
/// The parser buffers incoming text to handle tags that may arrive split across
/// multiple chunks. Call `push()` for each incoming chunk, then `flush()` when
/// the stream ends.
pub struct ThinkTagParser {
    in_think: bool,
    buffer: String,
    enable_thinking: bool,
}

impl ThinkTagParser {
    /// Create a new parser.
    ///
    /// * `enable_thinking` — if false, thinking content is silently discarded.
    /// * `start_in_think` — set true when the template forces a `<think>` block
    ///   open at the start (e.g. some chat templates inject it).
    pub fn new(enable_thinking: bool, start_in_think: bool) -> Self {
        Self {
            in_think: start_in_think,
            buffer: String::new(),
            enable_thinking,
        }
    }

    /// Push a chunk of text into the parser and return any events that can be
    /// emitted immediately.
    pub fn push(&mut self, text: &str) -> Vec<ThinkTagEvent> {
        self.buffer.push_str(text);
        self.drain(false)
    }

    /// Flush any remaining buffered content as a final event.
    pub fn flush(&mut self) -> Vec<ThinkTagEvent> {
        self.drain(true)
    }

    fn drain(&mut self, is_end: bool) -> Vec<ThinkTagEvent> {
        let mut events = Vec::new();

        loop {
            if self.in_think {
                if let Some(pos) = self.buffer.find("</think>") {
                    let before = self.buffer[..pos].to_string();
                    if self.enable_thinking && !before.is_empty() {
                        events.push(ThinkTagEvent::Thinking(before));
                    }
                    self.in_think = false;
                    let mut after = self.buffer[pos + 8..].to_string();
                    if after.starts_with('\n') {
                        after = after[1..].to_string();
                    }
                    self.buffer = after;
                } else if is_end {
                    if self.enable_thinking && !self.buffer.is_empty() {
                        events.push(ThinkTagEvent::Thinking(std::mem::take(&mut self.buffer)));
                    } else {
                        self.buffer.clear();
                    }
                    break;
                } else {
                    // Keep a safety margin so we don't emit a partial `</think>` tag
                    let safe_len = self.safe_flush_len(8);
                    if safe_len > 0 {
                        let chunk = self.buffer[..safe_len].to_string();
                        if self.enable_thinking {
                            events.push(ThinkTagEvent::Thinking(chunk));
                        }
                        self.buffer.drain(..safe_len);
                    }
                    break;
                }
            } else {
                if let Some(pos) = self.buffer.find("<think>") {
                    let before = self.buffer[..pos].to_string();
                    if !before.is_empty() {
                        events.push(ThinkTagEvent::Text(before));
                    }
                    self.in_think = true;
                    let mut after = self.buffer[pos + 7..].to_string();
                    if after.starts_with('\n') {
                        after = after[1..].to_string();
                    }
                    self.buffer = after;
                } else if is_end {
                    if !self.buffer.is_empty() {
                        events.push(ThinkTagEvent::Text(std::mem::take(&mut self.buffer)));
                    }
                    break;
                } else {
                    let safe_len = self.safe_flush_len(7);
                    if safe_len > 0 {
                        let chunk = self.buffer[..safe_len].to_string();
                        events.push(ThinkTagEvent::Text(chunk));
                        self.buffer.drain(..safe_len);
                    }
                    break;
                }
            }
        }

        events
    }

    /// Calculate how many bytes can be safely flushed without splitting a potential tag.
    fn safe_flush_len(&self, tag_len: usize) -> usize {
        let raw = self.buffer.len().saturating_sub(tag_len);
        // Ensure we don't split a multi-byte character
        let mut len = raw;
        while len > 0 && !self.buffer.is_char_boundary(len) {
            len -= 1;
        }
        len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_text() {
        let mut p = ThinkTagParser::new(true, false);
        let events = p.push("hello world");
        // Buffer is held because potential tag could span chunks
        assert!(events.is_empty() || events.iter().all(|e| matches!(e, ThinkTagEvent::Text(_))));
        let final_events = p.flush();
        let text: String = final_events
            .iter()
            .filter_map(|e| match e {
                ThinkTagEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("hello world"));
    }

    #[test]
    fn test_think_block() {
        let mut p = ThinkTagParser::new(true, false);
        let mut all = Vec::new();
        all.extend(p.push("<think>reasoning</think>answer"));
        all.extend(p.flush());
        assert!(all
            .iter()
            .any(|e| matches!(e, ThinkTagEvent::Thinking(t) if t.contains("reasoning"))));
        assert!(all
            .iter()
            .any(|e| matches!(e, ThinkTagEvent::Text(t) if t.contains("answer"))));
    }

    #[test]
    fn test_thinking_disabled() {
        let mut p = ThinkTagParser::new(false, false);
        let mut all = Vec::new();
        all.extend(p.push("<think>hidden</think>visible"));
        all.extend(p.flush());
        assert!(!all
            .iter()
            .any(|e| matches!(e, ThinkTagEvent::Thinking(_))));
        assert!(all
            .iter()
            .any(|e| matches!(e, ThinkTagEvent::Text(t) if t.contains("visible"))));
    }

    #[test]
    fn test_start_in_think() {
        let mut p = ThinkTagParser::new(true, true);
        let mut all = Vec::new();
        all.extend(p.push("already thinking</think>done"));
        all.extend(p.flush());
        assert!(all
            .iter()
            .any(|e| matches!(e, ThinkTagEvent::Thinking(t) if t.contains("already thinking"))));
        assert!(all
            .iter()
            .any(|e| matches!(e, ThinkTagEvent::Text(t) if t.contains("done"))));
    }
}
