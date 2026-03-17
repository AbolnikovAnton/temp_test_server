import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { OpenAI } from "openai";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// ✅ CORS для всех источников + preflight
app.use(cors());
app.options("*", cors());

// ✅ JSON парсер
app.use(express.json());

// ✅ OpenAI SDK
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ✅ Эндпоинт чата
app.post("/chat", async (req, res) => {
  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Некорректный формат messages" });
  }

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,
    });

    const reply =
      completion.choices?.[0]?.message?.content || "Ответ не получен";

    console.log("✅ Ответ:", reply.slice(0, 100) + "...");
    res.json({ reply });
  } catch (err) {
    console.error("❌ Ошибка OpenAI:", err.message || err);
    res.status(500).json({ error: "Ошибка при запросе к OpenAI" });
  }
});

// ✅ Запуск сервера
app.listen(port, () => {
  console.log(`🚀 Сервер запущен на http://localhost:${port}`);
});
