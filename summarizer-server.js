// summarizer-server.js — FAST OpenRouter summarizer (keep-alive + cache + bulk)
require('dotenv').config();
const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const cors = require('cors');
const crypto = require('crypto');
const httpMod = require('http');
const httpsMod = require('https');

const app = express();
app.use(bodyParser.json({ limit: '512kb' }));
app.use(cors({ origin: '*' })); // tighten in prod

const PORT = process.env.PORT || 5000;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

// ---------------- Tunables (speed-friendly defaults) ----------------
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || 'openai/gpt-4o-mini';
const SUMMARY_MAX_TOKENS = Number(process.env.SUMMARY_MAX_TOKENS || 120);

const MIN_WORDS = Number(process.env.SUMMARY_MIN_WORDS || 45);
const MAX_WORDS = Number(process.env.SUMMARY_MAX_WORDS || 60);
// Disable second pass expansion for latency (set to 1 if you insist on >=MIN_WORDS)
const EXPANSION_RETRIES = Number(process.env.SUMMARY_EXPANSION_RETRIES || 0);

// Pre-trim raw input to reduce cost/latency (news descriptions rarely need more)
const MAX_SOURCE_CHARS = Number(process.env.SUMMARY_MAX_SOURCE_CHARS || 1400);

// LRU cache size / TTL
const CACHE_CAP = Number(process.env.SUMMARY_CACHE_CAP || 600);
const CACHE_TTL_MS = Number(process.env.SUMMARY_CACHE_TTL_MS || 60 * 60 * 1000); // 1h

// Bulk concurrency to OpenRouter
const OPENROUTER_CONCURRENCY = Number(process.env.SUMMARY_OR_CONCURRENCY || 4);
// --------------------------------------------------------------------

// Keep-alive agents to reuse sockets (big win on Android devices)
const httpAgent = new httpMod.Agent({ keepAlive: true, maxSockets: 64 });
const httpsAgent = new httpsMod.Agent({ keepAlive: true, maxSockets: 64 });

const http = axios.create({
  baseURL: 'https://openrouter.ai/api/v1',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
  },
  timeout: 15000,
  httpAgent,
  httpsAgent,
});

// ---------------- Helpers ----------------
const now = () => Date.now();
const sha1 = (s) => crypto.createHash('sha1').update(String(s)).digest('hex');

function normalizeWhitespace(s) {
  return String(s || '')
    .replace(/[ \t\r\f\v]+/g, ' ')
    .replace(/\s*\n+\s*/g, ' ')
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

function preprocessSource(text) {
  if (!text) return '';
  const clean = String(text).slice(0, MAX_SOURCE_CHARS);
  return normalizeWhitespace(clean);
}

// ---------------- LRU cache with TTL + in-flight de-dup ----------------
const lru = new Map(); // key -> {summary, ts}
const inflight = new Map(); // key -> Promise<string>

function cacheGet(key) {
  const hit = lru.get(key);
  if (!hit) return null;
  if (now() - hit.ts > CACHE_TTL_MS) {
    lru.delete(key);
    return null;
  }
  // refresh LRU order
  lru.delete(key);
  lru.set(key, hit);
  return hit.summary;
}

function cacheSet(key, summary) {
  lru.set(key, { summary, ts: now() });
  if (lru.size > CACHE_CAP) {
    const first = lru.keys().next().value;
    if (first) lru.delete(first);
  }
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

function postProcess(raw) {
  // One short paragraph, no bullets/newlines; enforce <= MAX_WORDS
  let s = normalizeWhitespace(raw)
    .replace(/^\s*[-•\d]+\)?\.?\s*/g, ''); // strip accidental bullet prefix
  s = hardTrimToMaxWords(s, MAX_WORDS);
  return s;
}

async function summarizeOnce(source) {
  const system = [
    'You are a very fast news summarizer.',
    'Return ONE short paragraph (no line breaks).',
    `Keep it between ${MIN_WORDS}-${MAX_WORDS} words, crisp and easy to scan.`,
    'No headings, no quotes, no emojis.',
  ].join(' ');

  const msgs = [
    { role: 'system', content: system },
    { role: 'user', content: source },
  ];

  const out = await callOpenRouter(msgs);
  return postProcess(out);
}

async function summarizeWithOptionalExpand(source) {
  // First pass
  let s = await summarizeOnce(source);
  if (EXPANSION_RETRIES <= 0) return s;

  let words = countWords(s);
  let retries = EXPANSION_RETRIES;

  while (words < MIN_WORDS && retries > 0) {
    const system = [
      'Expand slightly while staying factual and concise.',
      `Return ONE paragraph (${MIN_WORDS}-${MAX_WORDS} words).`,
      'No bullets/headings/newlines.',
    ].join(' ');
    const msgs = [
      { role: 'system', content: system },
      { role: 'user', content: `SOURCE:\n${source}` },
      { role: 'user', content: `CURRENT:\n${s}` },
    ];
    s = postProcess(await callOpenRouter(msgs));
    words = countWords(s);
    retries--;
  }
  return s;
}

function pLimit(concurrency) {
  const queue = [];
  let active = 0;
  const next = () => {
    if (active >= concurrency) return;
    const job = queue.shift();
    if (!job) return;
    active++;
    job.fn().then(
      (v) => { active--; job.resolve(v); next(); },
      (e) => { active--; job.reject(e); next(); },
    );
  };
  return (fn) => new Promise((resolve, reject) => {
    queue.push({ fn, resolve, reject });
    next();
  });
}
const limitOR = pLimit(OPENROUTER_CONCURRENCY);

// ---------------- Routes ----------------
app.get('/', (_req, res) => res.json({ ok: true }));

app.post('/summarize', async (req, res) => {
  const text = req?.body?.text;
  const pre = preprocessSource(text);
  console.log(`→ /summarize ${new Date().toISOString()} chars=${(pre || '').length}`);

  if (!pre) return res.status(400).json({ error: 'Text is required' });

  const key = sha1(pre);
  const cached = cacheGet(key);
  if (cached) {
    console.log(`← /summarize 200 (cache) words≈${countWords(cached)}`);
    return res.json({ summary: cached, cached: true });
  }

  try {
    let promise = inflight.get(key);
    if (!promise) {
      promise = summarizeWithOptionalExpand(pre);
      inflight.set(key, promise);
    }
    const summary = await promise;
    inflight.delete(key);
    cacheSet(key, summary);
    console.log(`← /summarize 200 words≈${countWords(summary)}`);
    res.json({ summary });
  } catch (err) {
    inflight.delete(key);
    const status = err?.response?.status || 500;
    const data = err?.response?.data;
    console.error('❌ OpenRouter Error:', status, data || err.message);
    if (status === 402) {
      return res.status(502).json({ error: 'OpenRouter credits/limit', details: data?.error?.message });
    }
    res.status(500).json({ error: 'Failed to summarize', details: data || err.message });
  }
});

app.post('/summarize/bulk', async (req, res) => {
  const items = Array.isArray(req.body?.items) ? req.body.items : [];
  if (!items.length) return res.status(400).json({ error: 'items[] required' });

  // Build unique set by content hash (so identical texts share one OpenRouter call)
  const uniq = new Map(); // hash -> {ids:[], text}
  for (const it of items) {
    const id = String(it.id ?? '');
    const pre = preprocessSource(it.text);
    if (!id || !pre) continue;
    const h = sha1(pre);
    const entry = uniq.get(h) || { ids: [], text: pre };
    entry.ids.push(id);
    uniq.set(h, entry);
  }

  const out = {}; // id -> summary

  // 1) Fill from cache fast
  const toFetch = [];
  for (const [h, entry] of uniq.entries()) {
    const cached = cacheGet(h);
    if (cached) {
      for (const id of entry.ids) out[id] = cached;
    } else {
      toFetch.push({ hash: h, text: entry.text, ids: entry.ids });
    }
  }

  // 2) Fire off the rest with limited concurrency and in-flight coalescing
  await Promise.all(
    toFetch.map(({ hash, text, ids }) =>
      limitOR(async () => {
        try {
          let p = inflight.get(hash);
          if (!p) {
            p = summarizeWithOptionalExpand(text);
            inflight.set(hash, p);
          }
          const s = await p;
          inflight.delete(hash);
          cacheSet(hash, s);
          for (const id of ids) out[id] = s;
        } catch (e) {
          inflight.delete(hash);
        }
      })
    )
  );

  res.json({ summaries: out });
});

app.listen(PORT, () => {
  console.log(`✅ Summarizer running on ${PORT} — model=${SUMMARY_MODEL}`);
});
