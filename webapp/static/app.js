/**
 * app.js -- Scholar Stream frontend JavaScript
 *
 * Provides:
 *   - Async search via /api/search (fetch, no page reload)
 *   - Loading spinner state management
 *   - Copy-to-clipboard for generated answers
 *   - Slider live value display
 * 
 * Note: The main search form still submits via POST /search (server-rendered)
 * for full results. This file provides additional async UX enhancements.
 */

'use strict';

// ── Slider live display ───────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Sync any range inputs that have a matching *Val span
  document.querySelectorAll('input[type="range"]').forEach(slider => {
    const valSpan = document.getElementById(slider.id + 'Val');
    if (valSpan) {
      slider.addEventListener('input', () => {
        valSpan.textContent = slider.value;
      });
    }
  });

  // Attach copy buttons
  document.querySelectorAll('[data-copy-target]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.copyTarget);
      if (target) copyToClipboard(target.innerText, btn);
    });
  });
});


// ── Copy to clipboard ─────────────────────────────────────────────────────────

/**
 * Copy text to clipboard and show brief feedback on the button.
 *
 * @param {string} text    - Text to copy
 * @param {Element} btn    - Button element to update
 */
function copyToClipboard(text, btn) {
  if (!navigator.clipboard) {
    // Fallback for older browsers
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity  = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  } else {
    navigator.clipboard.writeText(text).catch(console.error);
  }
  if (btn) {
    const original = btn.textContent;
    btn.textContent = 'Copied!';
    btn.disabled    = true;
    setTimeout(() => {
      btn.textContent = original;
      btn.disabled    = false;
    }, 2000);
  }
}

// Expose globally so inline onclick handlers can use it
window.copyAnswer = function () {
  const body = document.getElementById('answerBody');
  if (body) copyToClipboard(body.innerText, document.querySelector('.copy-btn'));
};


// ── Async instant search (used on index page for live preview) ────────────────

/**
 * Fetch retrieval results without LLM generation (fast preview).
 * Called when user pauses typing in the search box.
 *
 * @param {string} query - User query string
 * @param {number} topK  - Number of results to retrieve
 */
async function fetchPreview(query, topK = 3) {
  if (!query || query.trim().length < 5) return;
  try {
    const url    = `/api/search?query=${encodeURIComponent(query)}&top_k=${topK}&gen_on=false`;
    const resp   = await fetch(url);
    if (!resp.ok) return;
    const data   = await resp.json();
    renderPreview(data.chunks || []);
  } catch (e) {
    // Silently ignore network errors in preview mode
  }
}

/**
 * Render a lightweight preview of source titles below the search box.
 *
 * @param {Array} chunks - Retrieved chunk objects
 */
function renderPreview(chunks) {
  let preview = document.getElementById('searchPreview');
  if (!preview) {
    preview = document.createElement('div');
    preview.id = 'searchPreview';
    preview.style.cssText = [
      'position:absolute', 'background:#fff', 'border:1px solid #ddd',
      'border-radius:6px', 'box-shadow:0 4px 16px rgba(0,0,0,.1)',
      'z-index:50', 'width:100%', 'max-height:220px', 'overflow:auto',
      'top:100%', 'left:0', 'margin-top:4px',
    ].join(';');

    const container = document.querySelector('.search-row');
    if (container) {
      container.style.position = 'relative';
      container.appendChild(preview);
    }
  }

  if (!chunks.length) {
    preview.style.display = 'none';
    return;
  }

  preview.style.display = 'block';
  preview.innerHTML = chunks.map(c => `
    <div style="padding:.6rem 1rem; border-bottom:1px solid #f0f0f0; font-size:.88rem; cursor:pointer;"
         onmouseenter="this.style.background='#f5f8ff'"
         onmouseleave="this.style.background='#fff'">
      <span style="color:#1F4E79; font-weight:600;">${escapeHtml(c.title || c.source)}</span>
      <span style="color:#9CA3AF; font-size:.78rem; margin-left:.5rem;">score ${c.score.toFixed(3)}</span>
      <div style="color:#4B5563; margin-top:.2rem;">${escapeHtml((c.excerpt || '').slice(0, 100))}...</div>
    </div>
  `).join('');
}

/**
 * Escape HTML entities to prevent XSS in dynamically inserted content.
 *
 * @param {string} str - Raw string
 * @returns {string} Escaped string
 */
function escapeHtml(str) {
  if (typeof str !== 'string') return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Attach debounced preview to search input on index page
document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('queryInput');
  if (!input) return;

  let debounceTimer;
  input.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    const preview = document.getElementById('searchPreview');
    if (preview) preview.style.display = 'none';

    debounceTimer = setTimeout(() => {
      const topK = parseInt(document.getElementById('topk')?.value || '3');
      fetchPreview(input.value.trim(), Math.min(topK, 3));
    }, 400);  // 400ms debounce
  });

  // Hide preview when user submits or clicks away
  input.addEventListener('blur', () => {
    setTimeout(() => {
      const preview = document.getElementById('searchPreview');
      if (preview) preview.style.display = 'none';
    }, 200);
  });
  input.addEventListener('focus', () => {
    if (input.value.trim().length >= 5) {
      const preview = document.getElementById('searchPreview');
      if (preview) preview.style.display = 'block';
    }
  });
});