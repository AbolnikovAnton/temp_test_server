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
    // Tried in order. Google tracks free-tier quota separately PER MODEL, so
    // this isn't just a fallback for outages — it's extra free daily
    // capacity. A brand-new "-latest" alias tends to launch with a *smaller*
    // introductory free quota than established models, so it goes first for
    // quality but the list gives us somewhere to go once it's exhausted.
    models: (process.env.GEMINI_MODELS || "gemini-flash-latest,gemini-2.5-flash,gemini-2.5-flash-lite")
      .split(",")
      .map((model) => model.trim())
      .filter(Boolean),
    maxOutputTokens: Number(process.env.GEMINI_MAX_TOKENS || 2048),
  },
  // Groq and Cerebras are both wholly free-tier inference services (not
  // "free variant of a paid API" like OpenRouter) — a separate quota pool
  // from Gemini and OpenRouter, so they're here purely for redundancy: if
  // Gemini and OpenRouter both hit their limits at once (e.g. a burst of
  // users), these are two more independent places to try before failing.
  groq: {
    apiKey: process.env.GROQ_API_KEY?.trim(),
    baseUrl: "https://api.groq.com/openai/v1",
    model: process.env.GROQ_MODEL || "llama-3.3-70b-versatile",
    maxTokens: Number(process.env.GROQ_MAX_TOKENS || 2048),
  },
  cerebras: {
    apiKey: process.env.CEREBRAS_API_KEY?.trim(),
    baseUrl: "https://api.cerebras.ai/v1",
    model: process.env.CEREBRAS_MODEL || "gpt-oss-120b",
    maxTokens: Number(process.env.CEREBRAS_MAX_TOKENS || 2048),
  },
  openrouter: {
    apiKey: process.env.OPENAI_API_KEY?.trim(),
    baseUrl: process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
    // A ":free" OpenRouter model costs no credits, sidestepping the account
    // balance entirely (20 req/min, 50/day — or 1000/day after a one-time
    // $10 lifetime credit purchase, per OpenRouter's own docs). See
    // https://openrouter.ai/models?variant=free for the current list.
    model: process.env.OPENROUTER_MODEL || "z-ai/glm-5.2:free",
    maxTokens: Number(process.env.OPENROUTER_MAX_TOKENS || 2048),
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

// "High demand" 503s are usually gone within a second or two, so a short
// retry here rides them out instead of immediately burning the next
// provider's own, unrelated budget for a transient blip. Applies to every
// OpenAI-compatible provider too (not just Gemini's JSON shape), hence the
// bare \b503\b for a plain "<Provider> 503: ..." message.
const RETRYABLE_ERROR_PATTERN = /"code":\s*503|UNAVAILABLE|overloaded|\b503\b/i;

function isRetryableError(err) {
  return RETRYABLE_ERROR_PATTERN.test(err?.message || String(err));
}

async function withRetries(fn, { attempts = 3, baseDelayMs = 500 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt === attempts - 1 || !isRetryableError(err)) throw err;
      const delay = baseDelayMs * 2 ** attempt;
      console.warn(`⚠️ Retryable error, waiting ${delay}ms before retry ${attempt + 2}/${attempts}:`, err.message || err);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastErr;
}

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
  // Groq and Cerebras are used here purely as free-tier services — $0
  // regardless of model, so these two entries just need to exist for
  // whatever the configured default model is.
  "llama-3.3-70b-versatile": { input: 0, output: 0 },
  "gpt-oss-120b": { input: 0, output: 0 },
};

function estimateCost(model, usage) {
  if (!usage) return null;
  // Any OpenRouter ":free" variant costs $0 by definition — no need to keep
  // a pricing entry in sync for every free model someone might configure.
  if (model?.endsWith(":free")) return 0;

  const pricing = PRICING_USD_PER_MILLION_TOKENS[model];
  if (!pricing) return null;
  return (
    (usage.inputTokens / 1_000_000) * pricing.input +
    (usage.outputTokens / 1_000_000) * pricing.output
  );
}

// The full provider error (raw JSON, stack traces, etc.) is only useful in
// the server logs — showing it straight to the chat UI is both ugly and
// unhelpful. This maps known failure shapes to a short, human sentence;
// callers still log the untouched error alongside it for debugging.
const PROVIDER_LABELS = {
  gemini: "Gemini",
  groq: "Groq",
  cerebras: "Cerebras",
  openrouter: "OpenRouter",
};

function describeProviderError(providerName, err) {
  const label = PROVIDER_LABELS[providerName] || providerName;
  const msg = err?.message || String(err);

  if (/"code":\s*503|UNAVAILABLE|overloaded/i.test(msg)) {
    return `${label} is temporarily overloaded — try again in a moment`;
  }
  if (/"code":\s*402|insufficient.*credit|requires more credits/i.test(msg)) {
    return `${label} is out of credits`;
  }
  if (/"code":\s*429|rate.?limit|quota/i.test(msg)) {
    return `${label} rate limit or quota reached`;
  }
  if (/"code":\s*401|unauthorized|invalid.*api.?key/i.test(msg)) {
    return `${label} rejected the API key`;
  }
  if (/model.*not.*found|does not exist|unknown model|invalid model/i.test(msg)) {
    return `${label} rejected the configured model — check its model id is still valid`;
  }

  // Unknown failure shape: still surface the HTTP status if we have one —
  // our own errors are formatted as "<Provider> <status>: <body>" — instead
  // of a completely opaque "request failed". The full body is still in the
  // server logs via console.warn for real debugging.
  const statusMatch = msg.match(/^\S+\s+(\d{3}):/);
  return statusMatch ? `${label} request failed (HTTP ${statusMatch[1]})` : `${label} request failed`;
}

// Shared implementation for any provider exposing an OpenAI-compatible
// /chat/completions endpoint (OpenRouter, Groq, Cerebras, ...). `cfg` needs
// { apiKey, baseUrl, model, maxTokens }. `includeStreamUsage` sends
// stream_options: { include_usage: true } during streaming to get real
// token counts — OpenAI-compatible in *spec*, but not every provider
// actually implements it, and some reject unrecognized fields outright with
// a 400 instead of ignoring them. Off by default; only flip it on for a
// provider once you've confirmed it doesn't break the stream — usage just
// comes back null otherwise (already handled gracefully everywhere).
function makeOpenAICompatibleProvider(name, cfg, { includeStreamUsage = false } = {}) {
  return {
    name,
    enabled: Boolean(cfg.apiKey),
    request: async (messages) => {
      const response = await withRetries(() =>
        fetch(`${cfg.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${cfg.apiKey}`,
          },
          body: JSON.stringify({
            model: cfg.model,
            messages,
            max_tokens: cfg.maxTokens,
            temperature: 0.7,
          }),
        }).then(async (res) => {
          const body = await res.text();
          if (!res.ok) throw new Error(`${name} ${res.status}: ${body}`);
          return body;
        }),
      );

      const data = JSON.parse(response);
      const text = data.choices?.[0]?.message?.content;
      if (!text) {
        throw new Error(`${name} returned no text`);
      }

      const usage = data.usage
        ? {
            inputTokens: data.usage.prompt_tokens || 0,
            outputTokens: data.usage.completion_tokens || 0,
          }
        : null;

      return { text, usage, model: cfg.model };
    },
    stream: async (messages, onDelta) => {
      // Only the call that opens the stream is retried, same reasoning as
      // Gemini's stream() — see the comment there.
      const response = await withRetries(() =>
        fetch(`${cfg.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${cfg.apiKey}`,
          },
          body: JSON.stringify({
            model: cfg.model,
            messages,
            max_tokens: cfg.maxTokens,
            temperature: 0.7,
            stream: true,
            ...(includeStreamUsage ? { stream_options: { include_usage: true } } : {}),
          }),
        }).then(async (res) => {
          if (!res.ok) {
            const body = await res.text();
            throw new Error(`${name} ${res.status}: ${body}`);
          }
          return res;
        }),
      );

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
        throw new Error(`${name} streamed no text`);
      }

      return { usage, model: cfg.model };
    },
  };
}

const providers = [
  {
    name: "gemini",
    enabled: Boolean(config.gemini.apiKey),
    request: async (messages) => {
      const prompt = messages
        .map((message) => {
          const role = message.role === "assistant" ? "Assistant" : message.role === "user" ? "User" : message.role;
          return `${role}: ${message.content}`;
        })
        .join("\n");

      const modelErrors = [];

      for (const model of config.gemini.models) {
        try {
          const response = await withRetries(() =>
            ai.models.generateContent({
              model,
              contents: prompt,
              config: { maxOutputTokens: config.gemini.maxOutputTokens },
            }),
          );

          const text = response.text || response.output?.[0]?.content?.[0]?.text;
          if (!text) {
            throw new Error(`no text in response: ${JSON.stringify(response)}`);
          }

          const usageMeta = response.usageMetadata;
          const usage = usageMeta
            ? {
                inputTokens: usageMeta.promptTokenCount || 0,
                outputTokens: usageMeta.candidatesTokenCount || 0,
              }
            : null;

          return { text, usage, model };
        } catch (err) {
          modelErrors.push(`${model}: ${err.message || err}`);
          console.warn(`⚠️ Gemini model "${model}" failed, trying next:`, err.message || err);
        }
      }

      throw new Error(modelErrors.join(" | "));
    },
    stream: async (messages, onDelta) => {
      const prompt = messages
        .map((message) => {
          const role = message.role === "assistant" ? "Assistant" : message.role === "user" ? "User" : message.role;
          return `${role}: ${message.content}`;
        })
        .join("\n");

      const modelErrors = [];

      for (const model of config.gemini.models) {
        let startedForThisModel = false;
        try {
          // Only the call that opens the stream is retried — once tokens
          // have started reaching the client (below), a later failure ends
          // the whole attempt instead of silently switching models or
          // retrying with a fresh (and possibly duplicate) response.
          const responseStream = await withRetries(() =>
            ai.models.generateContentStream({
              model,
              contents: prompt,
              config: { maxOutputTokens: config.gemini.maxOutputTokens },
            }),
          );

          let usage = null;
          let gotAnyText = false;

          for await (const chunk of responseStream) {
            if (chunk.text) {
              gotAnyText = true;
              startedForThisModel = true;
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
            throw new Error("streamed no text");
          }

          return { usage, model };
        } catch (err) {
          modelErrors.push(`${model}: ${err.message || err}`);
          if (startedForThisModel) {
            console.warn(`⚠️ Gemini model "${model}" failed mid-stream:`, err.message || err);
            throw err;
          }
          console.warn(`⚠️ Gemini model "${model}" failed, trying next:`, err.message || err);
        }
      }

      throw new Error(modelErrors.join(" | "));
    },
  },
  makeOpenAICompatibleProvider("groq", config.groq),
  makeOpenAICompatibleProvider("cerebras", config.cerebras),
  // Confirmed working via live testing — OpenRouter does return usage this way.
  makeOpenAICompatibleProvider("openrouter", config.openrouter, { includeStreamUsage: true }),
];

const getReply = async (messages) => {
  const friendlyErrors = [];

  for (const provider of providers.filter((p) => p.enabled)) {
    try {
      const { text, usage, model } = await provider.request(messages);
      return {
        text,
        provider: provider.name,
        model,
        usage,
        cost: estimateCost(model, usage),
      };
    } catch (err) {
      friendlyErrors.push(describeProviderError(provider.name, err));
      console.warn(`⚠️ ${provider.name} fallback error:`, err.message || err);
    }
  }

  throw new Error(friendlyErrors.length ? friendlyErrors.join(" · ") : "No provider is configured");
};

// Streams SSE events to the client as text arrives: {type:"chunk", text}
// while generating, then one {type:"done", ...} or {type:"error", ...}.
// Fallback to the next provider only works before any text has reached the
// client for this turn — once a provider has streamed something, switching
// providers mid-reply would just confuse the conversation, so a later
// failure ends the stream with an error instead of retrying silently.
const streamReply = async (messages, send) => {
  const friendlyErrors = [];

  for (const provider of providers.filter((p) => p.enabled)) {
    let startedStreaming = false;
    try {
      const { usage, model } = await provider.stream(messages, (delta) => {
        startedStreaming = true;
        send({ type: "chunk", text: delta });
      });
      send({
        type: "done",
        provider: provider.name,
        model,
        usage,
        cost: estimateCost(model, usage),
      });
      return;
    } catch (err) {
      const friendly = describeProviderError(provider.name, err);
      friendlyErrors.push(friendly);
      console.warn(`⚠️ ${provider.name} streaming fallback error:`, err.message || err);
      if (startedStreaming) {
        send({ type: "error", error: `${friendly} — reply cut short` });
        return;
      }
    }
  }

  send({ type: "error", error: friendlyErrors.length ? friendlyErrors.join(" · ") : "No provider is configured" });
};

if (!config.gemini.apiKey) {
  console.warn("⚠️ GEMINI_API_KEY is not configured — Gemini will be skipped.");
}
if (!config.groq.apiKey) {
  console.warn("⚠️ GROQ_API_KEY is not configured — Groq will be skipped.");
}
if (!config.cerebras.apiKey) {
  console.warn("⚠️ CEREBRAS_API_KEY is not configured — Cerebras will be skipped.");
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
