// summarizer-server.js — OpenRouter summarizer with min-words + fixed lines
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
const SUMMARY_MAX_TOKENS = Number(process.env.SUMMARY_MAX_TOKENS || 200);

// Desired words & lines
const TARGET_WORDS = Number(process.env.SUMMARY_TARGET_WORDS || 80);
const MIN_WORDS = Number(process.env.SUMMARY_MIN_WORDS || 70);
const MAX_WORDS = Number(process.env.SUMMARY_MAX_WORDS || 85);
const LINES = Number(process.env.SUMMARY_LINES || 8);

// If first pass < MIN_WORDS, retry once with an "expand" prompt
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
  return String(s || '').replace(/[ \t\r\f\v]+/g, ' ').replace(/\s*\n\s*/g, '\n').trim();
}

function countWords(s) {
  return normalizeWhitespace(s).split(/\s+/).filter(Boolean).length;
}

function hardTrimToMaxWords(s, max) {
  const words = normalizeWhitespace(s).split(/\s+/).filter(Boolean);
  if (words.length <= max) return words.join(' ');
  return words.slice(0, max).join(' ');
}

function ensureExactLineCountByReflow(text, lineCount) {
  // If the model didn't return exactly LINES, reflow words evenly.
  const words = normalizeWhitespace(text).split(/\s+/).filter(Boolean);
  if (words.length === 0) return ''.padEnd(lineCount - 1, '\n');

  const perLine = Math.ceil(words.length / lineCount);
  const lines = [];
  for (let i = 0; i < words.length; i += perLine) {
    lines.push(words.slice(i, i + perLine).join(' '));
  }
  // Guarantee exactly lineCount lines
  while (lines.length < lineCount) lines.push('');
  if (lines.length > lineCount) lines.length = lineCount;
  return lines.join('\n');
}

function postProcessToConstraints(raw) {
  // 1) Normalize
  let summary = normalizeWhitespace(raw);

  // 2) Remove bullets/numbers if any slipped in
  summary = summary
    .split('\n')
    .map(l => l.replace(/^\s*[-•\d]+\)?\.?\s*/g, '').trim())
    .filter(l => l.length > 0)
    .join('\n');

  // 3) If it already has newlines, keep them; otherwise we’ll reflow later.
  //    First, enforce max words so we don't overshoot.
  summary = hardTrimToMaxWords(summary, MAX_WORDS);

  // 4) Count words & lines
  let words = countWords(summary);
  let lines = summary.split('\n');

  // 5) If fewer than LINES lines, reflow
  if (lines.length !== LINES) {
    summary = ensureExactLineCountByReflow(summary, LINES);
    words = countWords(summary);
  }

  // 6) If still below MIN_WORDS (rare after expand), we can't invent content here.
  //    We'll let the caller decide to retry. For now, return the best effort.
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
  const wordsPerLineLow = Math.max(7, Math.floor((TARGET_WORDS - 10) / LINES));
  const wordsPerLineHigh = Math.ceil((TARGET_WORDS + 10) / LINES);

  const system = [
    'You are a concise news summarizer.',
    `Write EXACTLY ${LINES} separate lines (use plain newlines).`,
    `Aim for ~${TARGET_WORDS} words total; roughly ${wordsPerLineLow}-${wordsPerLineHigh} words per line.`,
    'No bullets or numbering. No headings. Keep facts tight and readable.',
  ].join(' ');

  const messages = [
    { role: 'system', content: system },
    { role: 'user', content: String(text) },
  ];

  return await callOpenRouter(messages);
}

async function expandSummaryToMin(text, firstSummary) {
  const wordsPerLineLow = Math.max(7, Math.floor((TARGET_WORDS - 10) / LINES));
  const wordsPerLineHigh = Math.ceil((TARGET_WORDS + 10) / LINES);

  const system = [
    'Improve and EXPAND the given summary while staying faithful to the source.',
    `Return EXACTLY ${LINES} lines separated by newlines, no bullets or numbering.`,
    `Ensure AT LEAST ${MIN_WORDS} words total, aiming near ${TARGET_WORDS} words; roughly ${wordsPerLineLow}-${wordsPerLineHigh} words per line.`,
    'Keep it readable and concise; do not add speculation.',
  ].join(' ');

  const messages = [
    { role: 'system', content: system },
    { role: 'user', content: `SOURCE TEXT:\n${String(text)}` },
    { role: 'user', content: `CURRENT (TO EXPAND):\n${String(firstSummary)}` },
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
    let summary = postProcessToConstraints(raw);
    let words = countWords(summary);
    let lineCount = summary.split('\n').length;

    // Retry to meet MIN_WORDS if needed
    let retriesLeft = EXPANSION_RETRIES;
    while (words < MIN_WORDS && retriesLeft > 0) {
      const expanded = await expandSummaryToMin(text, summary);
      summary = postProcessToConstraints(expanded);
      words = countWords(summary);
      lineCount = summary.split('\n').length;
      retriesLeft--;
    }

    // Final safety: if we still somehow exceed MAX_WORDS (rare), hard trim & reflow
    if (words > MAX_WORDS) {
      summary = hardTrimToMaxWords(summary, MAX_WORDS);
      summary = ensureExactLineCountByReflow(summary, LINES);
      words = countWords(summary);
      lineCount = summary.split('\n').length;
    }

    console.log(`← /summarize 200 words≈${words} lines=${lineCount}`);
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
