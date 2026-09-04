import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const config = {
  gemini: {
    apiKey: process.env.GEMINI_API_KEY?.trim(),
    model: process.env.GEMINI_MODEL || "gemini-flash-latest",
    maxOutputTokens: Number(process.env.GEMINI_MAX_TOKENS || 1024),
  },
  openrouter: {
    apiKey: process.env.OPENAI_API_KEY?.trim(),
    baseUrl: process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
    model: process.env.OPENROUTER_MODEL || "gpt-4o",
    maxTokens: Number(process.env.OPENROUTER_MAX_TOKENS || 1024),
  },
};

// Origins allowed to call this API. The client is a static site with no
// secrets of its own, so this (plus the rate limiter below) is a best-effort
// filter against casual/opportunistic abuse of the paid AI providers behind
// this server — not a strong auth boundary, since Origin can be spoofed by a
// non-browser client. It's the right tradeoff for a personal project.
const DEFAULT_ALLOWED_ORIGINS = ["https://abolnikovanton.github.io"];
const envOrigins = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);
const allowedOrigins = [...DEFAULT_ALLOWED_ORIGINS, ...envOrigins];

function isLocalhostOrigin(origin) {
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin);
}

function isAllowedOrigin(origin) {
  if (!origin) return false;
  return allowedOrigins.includes(origin) || isLocalhostOrigin(origin);
}

const corsOptions = {
  origin: (origin, callback) => {
    if (isAllowedOrigin(origin)) return callback(null, true);
    return callback(new Error("Not allowed by CORS"));
  },
};

const chatLimiter = rateLimit({
  windowMs: Number(process.env.RATE_LIMIT_WINDOW_MS || 15 * 60 * 1000),
  max: Number(process.env.RATE_LIMIT_MAX || 60),
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests, please try again later." },
});

const ai = new GoogleGenAI({ apiKey: config.gemini.apiKey });

// Best-effort snapshot of $ / 1M tokens, keyed by the exact model id each
// provider is configured with. Prices move over time (e.g. "gemini-flash-latest"
// is an alias Google repoints to a new model+price periodically) — treat this
// as an approximation, not a billing-grade source of truth, and refresh it
// when providers change their pricing. Unknown models return a null cost
// rather than silently charging the wrong price.
const PRICING_USD_PER_MILLION_TOKENS = {
  "gemini-flash-latest": { input: 0.75, output: 3.75 },
  "gemini-2.5-flash": { input: 0.15, output: 1.25 },
  "gemini-2.5-flash-lite": { input: 0.1, output: 0.4 },
  "gpt-4o": { input: 2.5, output: 10 },
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
};

function estimateCost(model, usage) {
  const pricing = PRICING_USD_PER_MILLION_TOKENS[model];
  if (!pricing || !usage) return null;
  return (
    (usage.inputTokens / 1_000_000) * pricing.input +
    (usage.outputTokens / 1_000_000) * pricing.output
  );
}

const providers = [
  {
    name: "gemini",
    model: config.gemini.model,
    enabled: Boolean(config.gemini.apiKey),
    request: async (messages) => {
      const prompt = messages
        .map((message) => {
          const role = message.role === "assistant" ? "Assistant" : message.role === "user" ? "User" : message.role;
          return `${role}: ${message.content}`;
        })
        .join("\n");

      const response = await ai.models.generateContent({
        model: config.gemini.model,
        contents: prompt,
        config: { maxOutputTokens: config.gemini.maxOutputTokens },
      });

      const text = response.text || response.output?.[0]?.content?.[0]?.text;
      if (!text) {
        throw new Error(`Gemini returned no text. Response structure: ${JSON.stringify(response)}`);
      }

      const usageMeta = response.usageMetadata;
      const usage = usageMeta
        ? {
            inputTokens: usageMeta.promptTokenCount || 0,
            outputTokens: usageMeta.candidatesTokenCount || 0,
          }
        : null;

      return { text, usage };
    },
    stream: async (messages, onDelta) => {
      const prompt = messages
        .map((message) => {
          const role = message.role === "assistant" ? "Assistant" : message.role === "user" ? "User" : message.role;
          return `${role}: ${message.content}`;
        })
        .join("\n");

      const responseStream = await ai.models.generateContentStream({
        model: config.gemini.model,
        contents: prompt,
        config: { maxOutputTokens: config.gemini.maxOutputTokens },
      });

      let usage = null;
      let gotAnyText = false;

      for await (const chunk of responseStream) {
        if (chunk.text) {
          gotAnyText = true;
          onDelta(chunk.text);
        }
        if (chunk.usageMetadata) {
          usage = {
            inputTokens: chunk.usageMetadata.promptTokenCount || 0,
            outputTokens: chunk.usageMetadata.candidatesTokenCount || 0,
          };
        }
      }

      if (!gotAnyText) {
        throw new Error("Gemini streamed no text");
      }

      return { usage };
    },
  },
  {
    name: "openrouter",
    model: config.openrouter.model,
    enabled: Boolean(config.openrouter.apiKey),
    request: async (messages) => {
      const response = await fetch(`${config.openrouter.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.openrouter.apiKey}`,
        },
        body: JSON.stringify({
          model: config.openrouter.model,
          messages,
          max_tokens: config.openrouter.maxTokens,
          temperature: 0.7,
        }),
      });

      const body = await response.text();
      if (!response.ok) {
        throw new Error(`OpenRouter ${response.status}: ${body}`);
      }

      const data = JSON.parse(body);
      const text = data.choices?.[0]?.message?.content;
      if (!text) {
        throw new Error("OpenRouter returned no text");
      }

      const usage = data.usage
        ? {
            inputTokens: data.usage.prompt_tokens || 0,
            outputTokens: data.usage.completion_tokens || 0,
          }
        : null;

      return { text, usage };
    },
    stream: async (messages, onDelta) => {
      const response = await fetch(`${config.openrouter.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.openrouter.apiKey}`,
        },
        body: JSON.stringify({
          model: config.openrouter.model,
          messages,
          max_tokens: config.openrouter.maxTokens,
          temperature: 0.7,
          stream: true,
          stream_options: { include_usage: true },
        }),
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`OpenRouter ${response.status}: ${body}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let usage = null;
      let gotAnyText = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sepIndex;
        while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, sepIndex).trim();
          buffer = buffer.slice(sepIndex + 2);

          if (!rawEvent.startsWith("data:")) continue;
          const payload = rawEvent.slice(5).trim();
          if (payload === "[DONE]") continue;

          let json;
          try {
            json = JSON.parse(payload);
          } catch {
            continue;
          }

          const delta = json.choices?.[0]?.delta?.content;
          if (delta) {
            gotAnyText = true;
            onDelta(delta);
          }
          if (json.usage) {
            usage = {
              inputTokens: json.usage.prompt_tokens || 0,
              outputTokens: json.usage.completion_tokens || 0,
            };
          }
        }
      }

      if (!gotAnyText) {
        throw new Error("OpenRouter streamed no text");
      }

      return { usage };
    },
  },
];

const getReply = async (messages) => {
  const errors = [];

  for (const provider of providers.filter((p) => p.enabled)) {
    try {
      const { text, usage } = await provider.request(messages);
      return {
        text,
        provider: provider.name,
        model: provider.model,
        usage,
        cost: estimateCost(provider.model, usage),
      };
    } catch (err) {
      errors.push(`${provider.name}: ${err.message || err}`);
      console.warn(`⚠️ ${provider.name} fallback error:`, err.message || err);
    }
  }

  throw new Error(errors.length ? errors.join(" | ") : "No provider is configured");
};

// Streams SSE events to the client as text arrives: {type:"chunk", text}
// while generating, then one {type:"done", ...} or {type:"error", ...}.
// Fallback to the next provider only works before any text has reached the
// client for this turn — once a provider has streamed something, switching
// providers mid-reply would just confuse the conversation, so a later
// failure ends the stream with an error instead of retrying silently.
const streamReply = async (messages, send) => {
  const errors = [];

  for (const provider of providers.filter((p) => p.enabled)) {
    let startedStreaming = false;
    try {
      const { usage } = await provider.stream(messages, (delta) => {
        startedStreaming = true;
        send({ type: "chunk", text: delta });
      });
      send({
        type: "done",
        provider: provider.name,
        model: provider.model,
        usage,
        cost: estimateCost(provider.model, usage),
      });
      return;
    } catch (err) {
      errors.push(`${provider.name}: ${err.message || err}`);
      console.warn(`⚠️ ${provider.name} streaming fallback error:`, err.message || err);
      if (startedStreaming) {
        send({ type: "error", error: `${provider.name} failed mid-stream: ${err.message || err}` });
        return;
      }
    }
  }

  send({ type: "error", error: errors.length ? errors.join(" | ") : "No provider is configured" });
};

if (!config.gemini.apiKey) {
  console.warn("⚠️ GEMINI_API_KEY is not configured — Gemini will be skipped.");
}
if (!config.openrouter.apiKey) {
  console.warn("⚠️ OPENAI_API_KEY is not configured — OpenRouter will be skipped.");
}

app.use(cors(corsOptions));
app.options("/*splat", cors(corsOptions));
app.use(express.json());

app.post("/chat", chatLimiter, async (req, res) => {
  const { messages, stream } = req.body;
  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Invalid messages format" });
  }

  if (!stream) {
    try {
      const { text, provider, model, usage, cost } = await getReply(messages);
      console.log("✅ Reply:", text.slice(0, 100) + "...");
      res.json({ reply: text, provider, model, usage, cost });
    } catch (err) {
      console.error("❌ Chat error:", err.message || err);
      res.status(500).json({ error: "Error while requesting AI providers", details: err.message || err });
    }
    return;
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  const send = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

  try {
    await streamReply(messages, send);
  } catch (err) {
    console.error("❌ Streaming chat error:", err.message || err);
    send({ type: "error", error: err.message || String(err) });
  } finally {
    res.end();
  }
});

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  if (err && err.message === "Not allowed by CORS") {
    return res.status(403).json({ error: "Origin not allowed" });
  }
  console.error("❌ Unhandled error:", err);
  res.status(500).json({ error: "Internal server error" });
});

app.listen(port, () => {
  console.log(`🚀 Server started at http://localhost:${port}`);
});
