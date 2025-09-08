// summarizer-server.js — Render-ready OpenRouter summarizer
require('dotenv').config();
const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
app.use(bodyParser.json());
app.use(cors({ origin: '*' })); // tighten to your domains in production

const PORT = process.env.PORT || 5000;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || 'openai/gpt-4.1-nano';
const SUMMARY_MAX_TOKENS = Number(process.env.SUMMARY_MAX_TOKENS || 120);

app.get('/', (_req, res) => res.json({ ok: true })); // health check

app.post('/summarize', async (req, res) => {
  const { text } = req.body;
  console.log(`→ /summarize ${new Date().toISOString()} len=${(text || '').length}`);

  if (!text || String(text).trim().length === 0) {
    return res.status(400).json({ error: 'Text is required for summarization.' });
  }

  try {
    const r = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: SUMMARY_MODEL,
        messages: [
          { role: 'system', content: 'Summarize the user text in 1–3 concise sentences.' },
          { role: 'user', content: String(text) }
        ],
        max_tokens: SUMMARY_MAX_TOKENS,
        temperature: 0.2,
        top_p: 0.9
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        },
        timeout: 30000
      }
    );

    const summary = r?.data?.choices?.[0]?.message?.content || '';
    console.log(`← /summarize 200 len=${(summary || '').length}`);

    res.json({ summary });
  } catch (err) {
    const status = err?.response?.status || 500;
    const data = err?.response?.data;
    console.error('❌ OpenRouter Error:', status, data || err.message);

    if (status === 402) {
      return res.status(502).json({
        error: 'OpenRouter credits/limit error. Reduce max_tokens or add credits.',
        details: data?.error?.message || 'Requires more credits or fewer tokens.'
      });
    }
    res.status(500).json({ error: 'Failed to summarize', details: data || err.message });
  }
});

app.listen(PORT, () => {
  console.log(`✅ OpenRouter summarizer running on port ${PORT}`);
});
