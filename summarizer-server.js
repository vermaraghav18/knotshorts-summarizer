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

// Model & length controls
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || 'openai/gpt-4.1-nano';
// 80 words ≈ 110–130 tokens. Give a little headroom.
const SUMMARY_MAX_TOKENS = Number(process.env.SUMMARY_MAX_TOKENS || 160);

// ----- helpers -----
function normalizeWhitespace(s) {
  return String(s || '').replace(/\s+/g, ' ').trim();
}

function trimToWords(text, maxWords) {
  const words = normalizeWhitespace(text).split(' ');
  return words.slice(0, maxWords).join(' ');
}

function splitIntoLinesByWords(text, targetLines) {
  const words = normalizeWhitespace(text).split(' ');
  if (words.length === 0) return '';

  const lines = Math.max(1, targetLines);
  const chunkSize = Math.ceil(words.length / lines);

  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(' '));
  }
  return chunks.join('\n');
}

// -------------------

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
          {
            role: 'system',
            content:
              // Ask for ~80 words, formatted as 7–8 plain lines (no bullets/numbers).
              'You are a concise summarizer. Return ~80 words split into 7–8 short lines. Each line should be a simple sentence or phrase. Use plain newlines between lines. Do not add numbering, bullets, headings, or extra commentary.'
          },
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

    // Raw model output
    let summary = r?.data?.choices?.[0]?.message?.content || '';
    summary = normalizeWhitespace(summary);

    // Enforce ~80 words hard cap
    summary = trimToWords(summary, 80);

    // Enforce 7–8 lines (default to 8). If fewer words, you'll naturally get fewer lines.
    summary = splitIntoLinesByWords(summary, 8);

    console.log(`← /summarize 200 words≈${summary.split(/\s+/).filter(Boolean).length} lines=${summary.split('\n').length}`);

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
