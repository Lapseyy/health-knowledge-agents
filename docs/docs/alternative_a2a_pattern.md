# Alternative: A2A Protocol Integration

Assignment 2 criterion 6 accepts either A2A protocol integration or MCP tool sharing. The reference implementation demonstrates MCP sharing through Tavily; students may instead choose A2A.

A qualifying A2A path should demonstrate at least:

- two distinct agents participating;
- Agent Card discovery;
- message/task exchange using the current A2A protocol surface;
- task lifecycle/status evidence;
- streaming/progress behavior where applicable;
- inspectable A2A traffic in the application trace;
- architecture rationale in `docs/architecture.md`.

A generic HTTP endpoint with agent-themed JSON but no A2A discovery/task semantics does not satisfy the protocol path.

Verify the current A2A specification and SDK before implementation; do not copy stale method names from this sketch.
