import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { OpenAI } from "openai";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// ✅ CORS for all origins + preflight
app.use(cors());
app.options("*", cors());

// ✅ JSON parser
app.use(express.json());

// ✅ OpenAI SDK
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
  defaultHeaders: {
    "HTTP-Referer": "https://abolnikovanton.github.io/temp_test_client/",
    "X-Title": "Anton Abolnikov CoPilot",
  },
});

// ✅ Chat endpoint
app.post("/chat", async (req, res) => {
  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Invalid messages format" });
  }

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,
    });

    const reply =
      completion.choices?.[0]?.message?.content || "No response received";

    console.log("✅ Reply:", reply.slice(0, 100) + "...");
    res.json({ reply });
  } catch (err) {
    console.error("❌ OpenAI error:", err.message || err);
    res.status(500).json({ error: "Error while requesting OpenAI" });
  }
});

// ✅ Start server
app.listen(port, () => {
  console.log(`🚀 Server started at http://localhost:${port}`);
});
