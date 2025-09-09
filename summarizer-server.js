// summarizer-server.js — OpenRouter summarizer: single paragraph, min words
require('dotenv').config();
const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
app.use(bodyParser.json());
app.use(cors({ origin: '*' })); // tighten in production

const PORT = process.env.PORT || 5000;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

// ------------------ Tunables ------------------
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || 'openai/gpt-4.1-nano';
const SUMMARY_MAX_TOKENS = Number(process.env.SUMMARY_MAX_TOKENS || 220);

const TARGET_WORDS = Number(process.env.SUMMARY_TARGET_WORDS || 80);
const MIN_WORDS = Number(process.env.SUMMARY_MIN_WORDS || 70);
const MAX_WORDS = Number(process.env.SUMMARY_MAX_WORDS || 95);

const EXPANSION_RETRIES = Number(process.env.SUMMARY_EXPANSION_RETRIES || 1);
// ------------------------------------------------

const http = axios.create({
  baseURL: 'https://openrouter.ai/api/v1',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
  },
  timeout: 30000,
});

// ---------- helpers ----------
function normalizeWhitespace(s) {
  return String(s || '').replace(/[ \t\r\f\v]+/g, ' ').replace(/\s*\n+\s*/g, ' ').trim();
}

function countWords(s) {
  return normalizeWhitespace(s).split(/\s+/).filter(Boolean).length;
}

function hardTrimToMaxWords(s, max) {
  const words = normalizeWhitespace(s).split(/\s+/).filter(Boolean);
  if (words.length <= max) return words.join(' ');
  return words.slice(0, max).join(' ');
}

function postProcessOneParagraph(raw) {
  // 1) normalize to one line
  let summary = normalizeWhitespace(raw);

  // 2) strip bullets/numbering if any slipped in
  summary = summary.replace(/(^|\s)[-•]\s+/g, ' ')
                   .replace(/\s*\d+\)\s+/g, ' ')
                   .replace(/\s*\d+\.\s+/g, ' ')
                   .replace(/\s+/g, ' ')
                   .trim();

  // 3) cap at MAX_WORDS if overshot
  summary = hardTrimToMaxWords(summary, MAX_WORDS);
  return summary;
}

async function callOpenRouter(messages) {
  const { data } = await http.post('/chat/completions', {
    model: SUMMARY_MODEL,
    messages,
    max_tokens: SUMMARY_MAX_TOKENS,
    temperature: 0.2,
    top_p: 0.9,
  });
  return data?.choices?.[0]?.message?.content || '';
}

async function generateSummaryFirstPass(text) {
  const system =
    `You are a precise news summarizer. Write ONE single-paragraph summary (~${TARGET_WORDS} words). ` +
    `Absolutely no line breaks, bullets, or headings. Output plain text only. Be faithful to the source.`;

  const messages = [
    { role: 'system', content: system },
    { role: 'user', content: String(text) },
  ];

  return await callOpenRouter(messages);
}

async function expandSummaryToMin(text, current) {
  const system =
    `Improve and EXPAND the summary into ONE single paragraph. ` +
    `Ensure AT LEAST ${MIN_WORDS} words, aiming near ${TARGET_WORDS}. ` +
    `No line breaks, bullets, or headings. Output plain text only. Stay factual.`;

  const messages = [
    { role: 'system', content: system },
    { role: 'user', content: `SOURCE TEXT:\n${String(text)}` },
    { role: 'user', content: `CURRENT SUMMARY (TO EXPAND):\n${String(current)}` },
  ];

  return await callOpenRouter(messages);
}

// ----------------------------------------------

app.get('/', (_req, res) => res.json({ ok: true })); // health check

app.post('/summarize', async (req, res) => {
  const { text } = req.body;
  console.log(`→ /summarize ${new Date().toISOString()} len=${(text || '').length}`);

  if (!text || String(text).trim().length === 0) {
    return res.status(400).json({ error: 'Text is required for summarization.' });
  }

  try {
    // First attempt
    let raw = await generateSummaryFirstPass(text);
    let summary = postProcessOneParagraph(raw);
    let words = countWords(summary);

    // Retry to meet MIN_WORDS if needed
    let retriesLeft = EXPANSION_RETRIES;
    while (words < MIN_WORDS && retriesLeft > 0) {
      const expanded = await expandSummaryToMin(text, summary);
      summary = postProcessOneParagraph(expanded);
      words = countWords(summary);
      retriesLeft--;
    }

    // Final safety: hard trim if still too long
    if (words > MAX_WORDS) {
      summary = hardTrimToMaxWords(summary, MAX_WORDS);
      words = countWords(summary);
    }

    console.log(`← /summarize 200 words≈${words}`);
    return res.json({ summary });
  } catch (err) {
    const status = err?.response?.status || 500;
    const data = err?.response?.data;
    console.error('❌ OpenRouter Error:', status, data || err.message);

    if (status === 402) {
      return res.status(502).json({
        error: 'OpenRouter credits/limit error. Reduce max_tokens or add credits.',
        details: data?.error?.message || 'Requires more credits or fewer tokens.',
      });
    }
    return res.status(500).json({ error: 'Failed to summarize', details: data || err.message });
  }
});

app.listen(PORT, () => {
  console.log(`✅ OpenRouter summarizer running on port ${PORT}`);
});
