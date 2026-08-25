import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const config = {
  gemini: {
    apiKey: process.env.GEMINI_API_KEY?.trim(),
    model: process.env.GEMINI_MODEL || "gemini-flash-latest",
  },
  openrouter: {
    apiKey: process.env.OPENAI_API_KEY?.trim(),
    baseUrl: process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
    model: process.env.OPENROUTER_MODEL || "gpt-4o",
    maxTokens: Number(process.env.OPENROUTER_MAX_TOKENS || 1024),
  },
};

const ai = new GoogleGenAI({ apiKey: config.gemini.apiKey });

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

      const response = await ai.models.generateContent({
        model: config.gemini.model,
        contents: prompt,
      });

      const text = response.text || response.output?.[0]?.content?.[0]?.text;
      if (!text) {
        throw new Error(`Gemini returned no text. Response structure: ${JSON.stringify(response)}`);
      }
      return text;
    },
  },
  {
    name: "openrouter",
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
      return text;
    },
  },
];

const getReply = async (messages) => {
  const errors = [];

  for (const provider of providers.filter((p) => p.enabled)) {
    try {
      return await provider.request(messages);
    } catch (err) {
      errors.push(`${provider.name}: ${err.message || err}`);
      console.warn(`⚠️ ${provider.name} fallback error:`, err.message || err);
    }
  }

  throw new Error(errors.length ? errors.join(" | ") : "No provider is configured");
};

if (!config.gemini.apiKey) {
  console.warn("⚠️ GEMINI_API_KEY is not configured — Gemini will be skipped.");
}
if (!config.openrouter.apiKey) {
  console.warn("⚠️ OPENAI_API_KEY is not configured — OpenRouter will be skipped.");
}

app.use(cors());
app.options("/*splat", cors());
app.use(express.json());

app.post("/chat", async (req, res) => {
  const { messages } = req.body;
  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Invalid messages format" });
  }

  try {
    const reply = await getReply(messages);
    console.log("✅ Reply:", reply.slice(0, 100) + "...");
    res.json({ reply });
  } catch (err) {
    console.error("❌ Chat error:", err.message || err);
    res.status(500).json({ error: "Error while requesting AI providers", details: err.message || err });
  }
});

app.listen(port, () => {
  console.log(`🚀 Server started at http://localhost:${port}`);
});
