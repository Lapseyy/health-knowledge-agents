# Frontend Vibe-Coding Prompt

Use this as a starting prompt for Codex CLI or Claude Code if rebuilding the reference frontend.

> Build a single-page vanilla HTML/CSS/JavaScript chat UI for a FastAPI backend. POST JSON `{query, session_id}` to `/api/chat`. The response is `text/event-stream`; because this is POST, consume the stream with `fetch()` plus `ReadableStream`, not native `EventSource`. Parse SSE frames named `token`, `trace`, `citation`, `error`, and `done`. Render the answer incrementally. Render routing traces in a separate panel with visually distinct badges for `agent_route`, `llm_route`, `rag_retrieval`, and `mcp_tool_call`. Render citations in their own panel. Disable submit while a stream is active. Include Reset behavior. Use no frontend framework and no Node build step.

The generated frontend must remain inspectable and easy for TAs to run from static files served by FastAPI.
