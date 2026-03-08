# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`navi-agent` is a Rust crate within the Navi workspace — an AI agent framework that provides LLM integration, tool execution, and an agentic loop. It supports multiple LLM backends (OpenAI Chat, OpenAI Responses, Anthropic, local llama.cpp) and includes filesystem tools for coding assistance (read, write, edit, grep, glob, bash).

## Build & Run

```bash
# Build this crate
cargo build -p navi-agent

# Run the coding agent CLI (nanocode)
cargo run --bin nanocode -p navi-agent -- --api responses --baseurl http://HOST:PORT/v1

# Run the LLM test binary
cargo run --bin test_llm -p navi-agent -- --api-key KEY --prompt "Hello"

# Lint
cargo clippy -p navi-agent

# Format
cargo fmt -p navi-agent
```

The `nanocode` binary accepts `--api` flag with values: `responses` (OpenAI Responses API), `anthropic`, `chat` (OpenAI Chat), or `local` (llama.cpp).

## Architecture

### Two-Layer Design: New Architecture + Legacy

The crate has a **new modular architecture** alongside **legacy code** still in use:

**New architecture** (preferred for new work):
- `core/` — Foundational types: `AgentEvent`, `AgentError`, `AgentState`, `NaviMessage`, `LLMMessage`, `ContentBlock`
- `runtime/` — The stateless `agent_loop()` function that drives the LLM→Tool cycle. Takes an `LlmClient` trait, `ToolRegistry`, `ContextPipeline`, and `ContextStore`, returns an `AgentEventStream`
- `agent/` — `NaviAgent` (high-level wrapper) built via `AgentBuilder`, plus `NaviBotLlmClient` compatibility adapter in `compat.rs`
- `context/` — Context management: `ContextStore` (in-memory conversation history), `ContextPipeline` (transforms), `MessageBasedPruner`
- `tool/` — `NaviTool` trait, `ToolRegistry`, `ToolExecutor`, and filesystem tools in `fs_tools/` (BashTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool)
- `llm/` — Multi-backend LLM clients implementing `ChatClient` trait:
  - `api_client.rs` — `APIChatClient` that dispatches to format-specific serializers
  - `openai_chat.rs` / `openai_responses.rs` / `anthropic.rs` — Protocol serializers
  - `local_client.rs` — llama.cpp integration via `navi-llm`
  - `stateful_client.rs` — Wraps `ChatClient` to implement `LlmClient` (the runtime trait)
  - `traits.rs` — `ChatClient` trait (stream-based, supports tools and vision)

**Legacy code** (still wired into `navi-bot`):
- `agents/` — Pre-built agents: `schedule.rs`, `task.rs`, `schedule_tools.rs`
- `executor.rs` — `LocalExecutor` for script execution
- `multiplexer.rs` — `OutputMultiplexer` and legacy `AgentEvent`

### Key Traits

- **`ChatClient`** (`llm/traits.rs`) — Low-level LLM interface with `chat_stream()`, tool definitions, vision support
- **`LlmClient`** (`runtime/llm_client.rs`) — Runtime-level trait consumed by `agent_loop()`, wraps ChatClient with tool defs and context
- **`NaviTool`** (`tool/traits.rs`) — Tool interface with name, description, JSON schema, and async `execute()`
- **`ContextStore`** (`context/store.rs`) — Conversation history storage
- **`ContextTransform`** (`context/pipeline.rs`) — Message pre-processing before LLM calls

### Agent Loop Flow

`agent_loop()` in `runtime/loop_impl.rs` is the core engine:
1. Loads context from `ContextStore`, applies `ContextPipeline` transforms
2. Sends messages to LLM via `LlmClient`, streams response events
3. If LLM returns tool calls → executes via `ToolExecutor` → appends results → loops back
4. Emits `AgentEvent`s (TextDelta, ThinkingDelta, ToolCallStart, ToolResult, TurnComplete, etc.) on a channel
5. Runs in a background tokio task; caller consumes `AgentEventStream`

### LLM API Format Dispatch

`APIChatClient` uses an `APIFormat` enum to select serialization:
- `APIFormat::OpenAIChat` → `openai_chat.rs` serializer
- `APIFormat::OpenAIResponses` → `openai_responses.rs` serializer
- `APIFormat::Anthropic` → `anthropic.rs` serializer

All formats produce the same `ChatDelta` stream (Text, Thinking, ToolCall).

## Dependencies

- `navi-core` — Shared workspace types
- `navi-llm` — Local LLM inference (llama.cpp bindings)
- `rusqlite` — SQLite for agent persistence (schedule/task agents)
- `reqwest` + `eventsource-stream` — HTTP SSE streaming for API clients
- `rustyline` — REPL in nanocode binary
- `ignore` — .gitignore-aware file traversal for tools
