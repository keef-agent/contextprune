# ContextPrune + Vercel AI SDK

Vercel AI SDK's `createAnthropic` factory accepts a `baseURL` option. Pass the ContextPrune proxy URL and every AI SDK call routes through it.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
npm install ai @ai-sdk/anthropic
```

## Integration

```typescript
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';

// Route all calls through ContextPrune
const anthropic = createAnthropic({
  baseURL: 'http://localhost:8899',
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const { text } = await generateText({
  model: anthropic('claude-sonnet-4-6'),
  prompt: 'Explain the difference between synchronous and asynchronous programming.',
});

console.log(text);
```

## With streaming

```typescript
import { createAnthropic } from '@ai-sdk/anthropic';
import { streamText } from 'ai';

const anthropic = createAnthropic({ baseURL: 'http://localhost:8899' });

const result = streamText({
  model: anthropic('claude-sonnet-4-6'),
  prompt: 'Write a haiku about distributed systems.',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

## With tools

```typescript
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText, tool } from 'ai';
import { z } from 'zod';

const anthropic = createAnthropic({ baseURL: 'http://localhost:8899' });

const { text, toolCalls } = await generateText({
  model: anthropic('claude-sonnet-4-6'),
  tools: {
    getWeather: tool({
      description: 'Get the weather for a location',
      parameters: z.object({ city: z.string() }),
      execute: async ({ city }) => ({ temperature: 72, condition: 'sunny', city }),
    }),
  },
  prompt: 'What is the weather in Boston?',
});
```

## Next.js API route

```typescript
// app/api/chat/route.ts
import { createAnthropic } from '@ai-sdk/anthropic';
import { streamText } from 'ai';

const anthropic = createAnthropic({ baseURL: 'http://localhost:8899' });

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: anthropic('claude-sonnet-4-6'),
    messages,
  });

  return result.toDataStreamResponse();
}
```

## Notes

- `baseURL` supported in `@ai-sdk/anthropic >= 0.1.0`
- Streaming requests pass through unchanged; ContextPrune compression runs on non-streaming calls
- Tool schemas are filtered per turn — most effective with 10+ registered tools
- Stats logged to `~/.contextprune/stats.jsonl`

See real-world results in our benchmarks.
