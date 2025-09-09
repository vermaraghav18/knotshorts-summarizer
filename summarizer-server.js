// summarizer-server.js — Fast OpenRouter summarizer (single paragraph, 60–75 words)
// Features: HTTP keep-alive, input trimming, strict word window, lightweight in-memory cache.

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');
const crypto = require('crypto');
const httpMod = require('http');
const httpsMod = require('https');

const app = express();
app.use(bodyParser.json({ limit: '256kb' }));
app.use(cors({ origin: '*' })); // tighten for prod

const PORT = process.env.PORT || 5000;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

// ------------------ Tunables (you can also set these in .env) ------------------
const SUMMARY_MODEL      = process.env.SUMMARY_MODEL || 'openai/gpt-4o-mini'; // fast + concise
const SUMMARY_MAX_TOKENS = Number(process.env.SUMMARY_MAX_TOKENS || 110);     // keep small for speed

// Hard target window
const MIN_WORDS = Number(process.env.SUMMARY_MIN_WORDS || 60);
const MAX_WORDS = Number(process.env.SUMMARY_MAX_WORDS || 75);

// Trim input so we don’t waste tokens/time on huge articles
const INPUT_MAX_CHARS = Number(process.env.SUMMARY_INPUT_MAX_CHARS || 1400);

// Retry once to expand if the first pass undershoots MIN_WORDS
const EXPANSION_RETRIES = Number(process.env.SUMMARY_EXPANSION_RETRIES || 1);

// Keep-alive sockets → lower latency across many requests
const httpAgent  = new httpMod.Agent({ keepAlive: true, maxSockets: 128, keepAliveMsecs: 60_000 });
const httpsAgent = new httpsMod.Agent({ keepAlive: true, maxSockets: 128, keepAliveMsecs: 60_000 });

const http = axios.create({
  baseURL: 'https://openrouter.ai/api/v1',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
  },
  httpAgent,
  httpsAgent,
  timeout: 15000,
});

// ------------------ tiny in-memory cache (LRU-ish) ------------------
const CACHE_MAX = Number(process.env.SUMMARY_CACHE_MAX || 800);
const CACHE_TTL_MS = Number(process.env.SUMMARY_CACHE_TTL_MS || 24 * 60 * 60 * 1000);
const _cache = new Map(); // key -> { value, t }

function _cacheGet(key) {
  const hit = _cache.get(key);
  if (!hit) return null;
  if (Date.now() - hit.t > CACHE_TTL_MS) {
    _cache.delete(key);
    return null;
  }
  // refresh recency
  _cache.delete(key);
  _cache.set(key, hit);
  return hit.value;
}
function _cacheSet(key, value) {
  if (_cache.size >= CACHE_MAX) {
    const first = _cache.keys().next().value;
    if (first) _cache.delete(first);
  }
  _cache.set(key, { value, t: Date.now() });
}

function normalizeWhitespace(s) {
  return String(s || '')
    .replace(/[\r\n]+/g, ' ')   // no line breaks at all
    .replace(/[ \t\f\v]+/g, ' ') // collapse spaces/tabs
    .trim();
}
function countWords(s) {
  return normalizeWhitespace(s).split(/\s+/).filter(Boolean).length;
}
function hardTrimToMaxWords(s, max) {
  const words = normalizeWhitespace(s).split(/\s+/).filter(Boolean);
  if (words.length <= max) return words.join(' ');
  return words.slice(0, max).join(' ');
}
function sha1(s) {
  return crypto.createHash('sha1').update(s).digest('hex');
}

async function callOpenRouter(messages) {
  const { data } = await http.post('/chat/completions', {
    model: SUMMARY_MODEL,
    messages,
    max_tokens: SUMMARY_MAX_TOKENS,
    temperature: 0.2,
    top_p: 0.7,
  });
  return data?.choices?.[0]?.message?.content || '';
}

async function firstPass(text) {
  const system = [
    'You are a fast, precise news summarizer.',
    `Return ONE single paragraph of ${MIN_WORDS}-${MAX_WORDS} words.`,
    'Absolutely no line breaks, bullets, or numbering.',
    'Be clear, neutral, and compact; focus on what happened and why it matters.',
  ].join(' ');
  return await callOpenRouter([
    { role: 'system', content: system },
    { role: 'user', content: text },
  ]);
}

async function expandPass(text, current) {
  const system = [
    'Improve and EXPAND this summary slightly while staying faithful to the source.',
    `Return ONE single paragraph of at least ${MIN_WORDS} words but not more than ${MAX_WORDS}.`,
    'No line breaks, bullets, or lists.',
  ].join(' ');
  return await callOpenRouter([
    { role: 'system', content: system },
    { role: 'user', content: `SOURCE:\n${text}` },
    { role: 'user', content: `CURRENT:\n${current}` },
  ]);
}

function postProcess(raw) {
  // single paragraph only
  let s = normalizeWhitespace(raw);

  // if model slipped bullets/nums, strip leading markers
  s = s.replace(/(^|\s)[-•\d]+\)?\.?\s+/g, ' ');

  // enforce hard upper bound (fast and predictable)
  s = hardTrimToMaxWords(s, MAX_WORDS);

  // ensure lower bound only if we already expanded in caller
  return s;
}

// ----------------------------------------------
app.get('/', (_req, res) => res.json({ ok: true })); // health check

app.post('/summarize', async (req, res) => {
  let { text } = req.body;
  const rawLen = (text || '').length;
  console.log(`→ /summarize ${new Date().toISOString()} len=${rawLen}`);

  if (!text || String(text).trim().length === 0) {
    return res.status(400).json({ error: 'Text is required for summarization.' });
  }

  // Trim giant inputs to keep latency low
  if (text.length > INPUT_MAX_CHARS) text = text.slice(0, INPUT_MAX_CHARS);

  const key = sha1(`${SUMMARY_MODEL}|${MIN_WORDS}|${MAX_WORDS}|${text}`);
  const cached = _cacheGet(key);
  if (cached) {
    console.log(`← /summarize 200 (cache) words≈${countWords(cached)}`);
    return res.json({ summary: cached });
  }

  try {
    // First attempt
    let raw = await firstPass(text);
    let summary = postProcess(raw);
    let words = countWords(summary);

    // Expand if too short
    let retries = EXPANSION_RETRIES;
    while (words < MIN_WORDS && retries-- > 0) {
      const expanded = await expandPass(text, summary);
      summary = postProcess(expanded);
      words = countWords(summary);
    }

    // Safety: re-trim if the model overshot (rare with hardTrim)
    if (words > MAX_WORDS) {
      summary = hardTrimToMaxWords(summary, MAX_WORDS);
      words = countWords(summary);
    }

    _cacheSet(key, summary);
    console.log(`← /summarize 200 words≈${words}`);
    return res.json({ summary });
  } catch (err) {
    const status = err?.response?.status || 500;
    const data = err?.response?.data;
    console.error('❌ OpenRouter Error:', status, data || err.message);

    if (status === 402) {
      return res.status(502).json({
        error: 'OpenRouter credits/limit error. Add credits or reduce usage.',
        details: data?.error?.message || 'Requires more credits.',
      });
    }
    return res.status(500).json({ error: 'Failed to summarize', details: data || err.message });
  }
});

app.listen(PORT, () => {
  console.log(`✅ OpenRouter summarizer running on port ${PORT}`);
});
