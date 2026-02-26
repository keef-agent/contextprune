# ContextPrune + Mastra

Mastra is a TypeScript agent framework. Pass a custom Anthropic SDK instance with `baseURL` set to the ContextPrune proxy.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
npm install @mastra/core @anthropic-ai/sdk
```

## Integration

```typescript
import { Mastra } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import Anthropic from '@anthropic-ai/sdk';

// Route all Anthropic calls through ContextPrune
const anthropic = new Anthropic({
  baseURL: 'http://localhost:8899',
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const agent = new Agent({
  name: 'my-agent',
  instructions: 'You are a concise, accurate assistant.',
  model: {
    provider: 'ANTHROPIC',
    toolChoice: 'auto',
    name: 'claude-sonnet-4-6',
  },
  // Pass the custom client
  client: anthropic,
});

const mastra = new Mastra({ agents: { myAgent: agent } });

// Run the agent
const result = await mastra.getAgent('myAgent').generate(
  'Summarize the key differences between REST and GraphQL.'
);
console.log(result.text);
```

## With tools

```typescript
import { Mastra } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { createTool } from '@mastra/core/tools';
import Anthropic from '@anthropic-ai/sdk';
import { z } from 'zod';

const anthropic = new Anthropic({ baseURL: 'http://localhost:8899' });

const weatherTool = createTool({
  id: 'get_weather',
  description: 'Get the current weather for a city',
  inputSchema: z.object({ city: z.string() }),
  execute: async ({ context }) => {
    return { temperature: 72, condition: 'sunny', city: context.city };
  },
});

const agent = new Agent({
  name: 'weather-agent',
  instructions: 'Help users with weather information.',
  model: { provider: 'ANTHROPIC', name: 'claude-sonnet-4-6', toolChoice: 'auto' },
  tools: { weatherTool },
  client: anthropic,
});
```

## Notes

- `@anthropic-ai/sdk` `baseURL` option routes all requests through the proxy
- Tool-heavy Mastra agents benefit from ContextPrune's tool schema filtering (sends only relevant tools)
- Stats logged to `~/.contextprune/stats.jsonl`
- Streaming is passed through unchanged; deduplication runs on non-streaming requests

See real-world results in our benchmarks.
