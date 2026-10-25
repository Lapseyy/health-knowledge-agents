# Alternative: Vertex AI Search for Live Domain Retrieval

Tavily Remote MCP is the canonical live/current-knowledge integration.

Vertex AI Search is an accepted alternative **only when the project uses an externally updated domain-specific source that supplies current information**. A static Vertex AI Search datastore containing the same fixed corpus already used for RAG does not satisfy the separate live-knowledge requirement.

## Appropriate use case

Use Vertex AI Search when, for example, an organization's externally maintained documentation or catalog is continuously updated and the agent must retrieve from that changing domain.

## Architecture

```text
Research/Summary Agent
        |
        v
Vertex AI Search client
        |
        v
Externally updated domain datastore
```

Keep the local/static RAG corpus path logically separate unless the project intentionally chooses Pinecone/Vertex-managed storage for that role as well.

## Setup outline

1. Enable the current Vertex AI Search / Discovery Engine APIs.
2. Create and populate the datastore from the qualifying external/updating source.
3. Grant the Cloud Run service account only the roles required by the current API.
4. Add the required project, location, datastore, and engine identifiers to `env.example`.
5. Replace the Tavily live-retrieval adapter with a Vertex AI Search adapter and expose equivalent trace evidence.

Verify current Google Cloud commands, IAM roles, API names, and pricing at implementation time.
