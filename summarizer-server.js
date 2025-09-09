// summarizer-server.js — OpenRouter summarizer (STRICT 60–75 words, 1 paragraph)
require('dotenv').config();
const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
app.use(bodyParser.json());
app.use(cors({ origin: '*' })); // tighten to your domains in prod

const PORT = process.env.PORT || 5000;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

// ------------------ Tunables ------------------
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || 'openai/gpt-4.1-nano';
const SUMMARY_MAX_TOKENS = Number(process.env.SUMMARY_MAX_TOKENS || 180);

// HARD word range (inclusive)
const MIN_WORDS = Number(process.env.SUMMARY_MIN_WORDS || 60);
const MAX_WORDS = Number(process.env.SUMMARY_MAX_WORDS || 75);
const TARGET_WORDS = Math.round((MIN_WORDS + MAX_WORDS) / 2);

// If first pass < MIN_WORDS, retry once with an "expand" prompt
const EXPANSION_RETRIES = Number(process.env.SUMMARY_EXPANSION_RETRIES || 1);
// ------------------------------------------------

const http = axios.create({
  baseURL: 'https://openrouter.ai/api/v1',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${OPENROUTER_API_KEY}`,
  },
  timeout: 30000,
});

// ---------- helpers ----------
function normalizeWhitespace(s) {
  // collapse whitespace; also collapse newlines to spaces (single paragraph)
  return String(s || '')
    .replace(/[ \t\r\f\v]+/g, ' ')
    .replace(/\s*\n+\s*/g, ' ')
    .trim();
}

function stripBulletsAndMarkers(s) {
  return s
    .split('\n')
    .map(l => l.replace(/^\s*[-•\d]+\)?\.?\s*/g, '').trim())
    .filter(Boolean)
    .join(' ');
}

function countWords(s) {
  const words = normalizeWhitespace(s).split(/\s+/).filter(Boolean);
  return words.length;
}

function hardTrimToMaxWords(s, max) {
  const words = normalizeWhitespace(s).split(/\s+/).filter(Boolean);
  if (words.length <= max) return words.join(' ');
  // preserve punctuation by trimming on word boundary
  let out = words.slice(0, max).join(' ');
  // ensure we end cleanly
  out = out.replace(/[,:;—–-]+$/g, '').replace(/\s+$/g, '');
  if (!/[.!?]$/.test(out)) out += '.';
  return out;
}

function postProcess(raw) {
  // 1) normalize, remove bullets, force one paragraph
  let s = stripBulletsAndMarkers(normalizeWhitespace(raw));

  // 2) kill accidental headings/line breaks
  s = s.replace(/\s*\n+\s*/g, ' ');

  // 3) collapse multiple spaces
  s = s.replace(/\s{2,}/g, ' ').trim();

  return s;
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

async function generateFirstPass(text) {
  const system = [
    'You are a precise news summarizer.',
    `Write ONE paragraph between ${MIN_WORDS} and ${MAX_WORDS} words (strict).`,
    `Aim for about ${TARGET_WORDS} words.`,
    'No bullets, no lists, no headings, no line breaks; return plain text.',
    'Neutral, concise, factual; do not add speculation.',
  ].join(' ');

  const messages = [
    { role: 'system', content: system },
    { role: 'user', content: String(text) },
  ];

  return await callOpenRouter(messages);
}

async function expandSummary(text, current) {
  const system = [
    'Expand the following summary while staying strictly faithful to the source.',
    `Return ONE paragraph between ${MIN_WORDS} and ${MAX_WORDS} words.`,
    `Aim near ${TARGET_WORDS} words. No bullets, no lists, no line breaks.`,
    'Keep a clean, neutral news tone. No extra commentary.',
  ].join(' ');

  const messages = [
    { role: 'system', content: system },
    { role: 'user', content: `SOURCE:\n${String(text)}` },
    { role: 'user', content: `CURRENT SUMMARY:\n${String(current)}` },
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
    let raw = await generateFirstPass(text);
    let summary = postProcess(raw);
    let words = countWords(summary);

    // Retry to meet MIN_WORDS if needed
    let retriesLeft = EXPANSION_RETRIES;
    while (words < MIN_WORDS && retriesLeft > 0) {
      const expanded = await expandSummary(text, summary);
      summary = postProcess(expanded);
      words = countWords(summary);
      retriesLeft--;
    }

    // Enforce upper bound — never exceed MAX_WORDS
    if (words > MAX_WORDS) {
      summary = hardTrimToMaxWords(summary, MAX_WORDS);
      words = countWords(summary);
    }

    // Safety: if still below MIN (edge case), just return best effort (client will ellipsize)
    console.log(`← /summarize 200 words=${words}`);
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
