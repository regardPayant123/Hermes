# Hermes Tools

## Built-in tools

The tool registry exposes `current_time`, a restricted arithmetic `calculator`, `rag_search` when a RAG pipeline is supplied, and read-only `web_search`. Web search supports Tavily (`TAVILY_API_KEY`) and Brave (`BRAVE_SEARCH_API_KEY`); credentials are read from the environment, never from the JSON file.

Install MCP support separately when connecting external servers:

```bash
pip install -r requirements-tools.txt
```

## MCP lifecycle

Configure servers in `configs/tools.json`. Servers are disabled unless `enabled` is explicitly `true`. Supported transports are `stdio`, `streamable_http`, and `sse`. Remote tools are registered with a server-name prefix such as `filesystem.read_file` to avoid collisions. Environment values can reference process variables, for example `${MY_MCP_TOKEN}`.

For example, an explicitly trusted local server can be configured like this:

```json
{
    "name": "filesystem",
    "enabled": false,
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
}
```

Set `enabled` to `true` only after reviewing the server command, its arguments, and the filesystem/network access it grants. For remote servers, use `transport: "streamable_http"`, a trusted HTTPS `url`, and headers whose secret values come from environment variables.

Start the tool runtime inside an async context so sessions and child processes are closed cleanly. Inject an async `approval_callback` into `ToolFactory.from_json` to authorize MCP tools that are not marked read-only; without one, those calls are denied:

```python
from engines.brain.factory import BrainFactory
from engines.knowledge.factory import KnowledgeFactory
from engines.tools import ToolFactory

knowledge = KnowledgeFactory.from_json("configs/knowledge.json")
runtime = ToolFactory.from_json("configs/tools.json", rag=knowledge["rag"])

async with runtime:
    brain = BrainFactory.from_json("configs/brain.json", tools=runtime.registry)
    response = await brain.run("Search my notes and the web for this topic.")
```

MCP server commands and endpoints are trusted configuration: enabling a stdio server launches its configured process, and enabling an HTTP server connects to its configured endpoint. Do not enable unreviewed servers. Tool arguments are schema-validated, calls have timeouts, results are size-limited, and LangGraph limits repeated tool rounds.