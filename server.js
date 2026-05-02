import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const geminiApiKey = process.env.GEMINI_API_KEY;
const geminiBaseUrl = process.env.GEMINI_BASE_URL || "https://gemini.googleapis.com/v1";
const geminiModel = process.env.GEMINI_MODEL || "gemini-1.5-pro";

const openRouterApiKey = process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY;
const openRouterBaseUrl = process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";
const openRouterModel = process.env.OPENROUTER_MODEL || "gpt-4o";
const openRouterMaxTokens = Number(process.env.OPENROUTER_MAX_TOKENS || 1024);

const fallbackProviders = ["gemini", "openrouter"];

const buildPromptForGemini = (messages) =>
  messages
    .map((message) => {
      const role = message.role === "assistant" ? "Assistant" : message.role === "user" ? "User" : message.role;
      return `${role}: ${message.content}`;
    })
    .join("\n");

const getGeminiResponse = async (messages) => {
  if (!geminiApiKey) {
    throw new Error("Gemini API key is not configured");
  }

  const prompt = buildPromptForGemini(messages);
  const response = await fetch(`${geminiBaseUrl}/models/${geminiModel}:generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${geminiApiKey}`,
    },
    body: JSON.stringify({
      model: geminiModel,
      temperature: 0.7,
      maxOutputTokens: 1024,
      candidateCount: 1,
      input: { text: prompt },
    }),
  });

  const raw = await response.text();
  if (!response.ok) {
    throw new Error(`Gemini (${response.status}): ${raw}`);
  }

  const data = JSON.parse(raw);
  const text = data.candidates?.[0]?.output?.[0]?.content?.[0]?.text;
  if (!text) {
    throw new Error("Gemini returned no text candidate");
  }

  return text;
};

const getOpenRouterResponse = async (messages) => {
  if (!openRouterApiKey) {
    throw new Error("OpenRouter API key is not configured");
  }

  const response = await fetch(`${openRouterBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${openRouterApiKey}`,
    },
    body: JSON.stringify({
      model: openRouterModel,
      messages,
      max_tokens: openRouterMaxTokens,
      temperature: 0.7,
    }),
  });

  const raw = await response.text();
  if (!response.ok) {
    throw new Error(`OpenRouter (${response.status}): ${raw}`);
  }

  const data = JSON.parse(raw);
  const text = data.choices?.[0]?.message?.content;
  if (!text) {
    throw new Error("OpenRouter returned no message content");
  }

  return text;
};

const getReply = async (messages) => {
  const errors = [];

  for (const provider of fallbackProviders) {
    try {
      if (provider === "gemini") {
        return await getGeminiResponse(messages);
      }
      if (provider === "openrouter") {
        return await getOpenRouterResponse(messages);
      }
    } catch (err) {
      errors.push({ provider, message: err.message || err.toString() });
      console.warn(`⚠️ ${provider} fallback error:`, err.message || err);
    }
  }

  throw new Error(`All providers failed: ${errors.map((e) => `${e.provider}: ${e.message}`).join(" | ")}`);
};

if (!geminiApiKey) {
  console.warn("⚠️ GEMINI_API_KEY is not configured. Google Gemini will be skipped.");
}

if (!openRouterApiKey) {
  console.warn("⚠️ OPENROUTER_API_KEY / OPENAI_API_KEY is not configured. OpenRouter fallback will be skipped.");
}

// ✅ CORS for all origins + preflight
app.use(cors());
app.options("*", cors());

// ✅ JSON parser
app.use(express.json());

// ✅ Chat endpoint
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

// ✅ Start server
app.listen(port, () => {
  console.log(`🚀 Server started at http://localhost:${port}`);
});
