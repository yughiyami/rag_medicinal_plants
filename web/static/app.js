const API = '';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const form = $('#query-form');
const input = $('#query-input');
const submitBtn = $('#submit-btn');
const btnText = $('.btn-text');
const btnSpinner = $('.btn-spinner');
const backendSelect = $('#backend-select');
const resultsSection = $('#results-section');
const errorSection = $('#error-section');

// --- Init ---
async function init() {
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    const badge = $('#status-badge');
    badge.textContent = data.status === 'ok' ? 'Online' : 'Error';
    badge.style.background = data.status === 'ok' ? 'var(--green)' : 'var(--red)';

    $('#vectorstore-badge').textContent = `${data.vectorstore_size.toLocaleString()} vectors`;

    // Disable unavailable backends
    const options = backendSelect.options;
    for (let i = 0; i < options.length; i++) {
      const val = options[i].value;
      if (!data.backends[val]) {
        options[i].disabled = true;
        options[i].textContent += ' (unavailable)';
      }
    }
  } catch {
    $('#status-badge').textContent = 'Offline';
    $('#status-badge').style.background = 'var(--red)';
  }
}

// --- Query ---
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  setLoading(true);
  resultsSection.classList.add('hidden');
  errorSection.classList.add('hidden');

  try {
    const res = await fetch(`${API}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        backend: backendSelect.value,
        language: 'auto',
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

// --- Examples ---
$$('.example-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    input.value = btn.dataset.query;
    input.focus();
  });
});

// --- Render ---
function renderResults(data) {
  resultsSection.classList.remove('hidden');

  // Tags
  const cat = data.classification?.category || 'unknown';
  const tagCat = $('#tag-category');
  tagCat.textContent = cat;
  tagCat.className = `tag tag-${cat}`;

  const crag = data.crag_action || 'unknown';
  const tagCrag = $('#tag-crag');
  tagCrag.textContent = `CRAG: ${crag}`;
  tagCrag.className = `tag tag-${crag}`;

  $('#tag-model').textContent = data.model || '--';
  const latSec = (data.latency_ms / 1000).toFixed(1);
  $('#tag-latency').textContent = `⏱ ${latSec}s`;
  const tokensEl = $('#tag-tokens');
  if (tokensEl) tokensEl.textContent = `${data.tokens_generated || 0} tokens`;

  // Answer — highlight citation references
  const answerEl = $('#answer-text');
  let answerHtml = escapeHtml(data.answer || 'No answer generated.');
  answerHtml = answerHtml.replace(
    /\[(\d+)\]/g,
    '<span class="cite-ref" data-cite="$1">[$1]</span>'
  );
  answerEl.innerHTML = answerHtml;

  // Citations
  const citList = $('#citations-list');
  citList.innerHTML = '';
  const citations = data.citations || [];
  $('#citations-count').textContent = `(${citations.length})`;

  citations.forEach(c => {
    const div = document.createElement('div');
    div.className = 'citation-item';
    div.id = `cite-${c.index}`;

    let meta = [];
    if (c.authors?.length) meta.push(c.authors.slice(0, 3).join(', '));
    if (c.year) meta.push(c.year);
    if (c.source) meta.push(c.source);
    if (c.pmid) meta.push(`PMID: ${c.pmid}`);

    let speciesHtml = '';
    if (c.species?.length) {
      speciesHtml = c.species.map(s => `<span class="citation-species">${escapeHtml(s)}</span>`).join('');
    }

    div.innerHTML = `
      <div>
        <span class="citation-index">[${c.index}]</span>
        <span class="citation-title">${escapeHtml(c.title || 'Untitled')}</span>
        ${speciesHtml}
      </div>
      ${meta.length ? `<div class="citation-meta">${escapeHtml(meta.join(' | '))}</div>` : ''}
    `;
    citList.appendChild(div);
  });

  // Click citation refs to scroll
  $$('.cite-ref').forEach(ref => {
    ref.addEventListener('click', () => {
      const target = document.getElementById(`cite-${ref.dataset.cite}`);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });

  // Trace
  const traceList = $('#trace-list');
  traceList.innerHTML = '';
  (data.trace_summary || []).forEach((step, i) => {
    if (i > 0) {
      const arrow = document.createElement('span');
      arrow.className = 'trace-arrow';
      arrow.textContent = '→';
      traceList.appendChild(arrow);
    }
    const span = document.createElement('span');
    span.className = 'trace-step';
    span.textContent = step;
    traceList.appendChild(span);
  });
}

function showError(msg) {
  errorSection.classList.remove('hidden');
  $('#error-text').textContent = msg;
}

function setLoading(loading) {
  submitBtn.disabled = loading;
  btnText.classList.toggle('hidden', loading);
  btnSpinner.classList.toggle('hidden', !loading);
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// --- Boot ---
init();
