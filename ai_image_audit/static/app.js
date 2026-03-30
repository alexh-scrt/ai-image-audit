/**
 * app.js – AI Image Audit frontend application
 *
 * Drives the single-page dashboard:
 *  - Submits scan jobs via POST /api/scan
 *  - Renders image report cards with metadata panels
 *  - Fetches and displays suggestion carousels via GET /api/suggestions
 *  - Handles error states and loading indicators
 *
 * No external dependencies – vanilla ES2020.
 */

'use strict';

// ============================================================
// State
// ============================================================

/** @type {Array<Object>} All image report objects from the last scan */
let allReports = [];

/** @type {string} Current filter: 'all' | 'flagged' | 'safe' */
let currentFilter = 'all';

/** @type {string|null} Source string of the image whose suggestions are shown */
let activeSuggestionSource = null;

// ============================================================
// DOM references (populated after DOMContentLoaded)
// ============================================================

const dom = {};

// ============================================================
// Initialisation
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  dom.scanForm       = document.getElementById('scan-form');
  dom.targetInput    = document.getElementById('target-input');
  dom.thresholdInput = document.getElementById('threshold-input');
  dom.scanBtn        = document.getElementById('scan-btn');
  dom.formError      = document.getElementById('form-error');

  dom.progressPanel  = document.getElementById('progress-panel');
  dom.progressMsg    = document.getElementById('progress-message');

  dom.summaryBar     = document.getElementById('summary-bar');
  dom.statTotal      = document.getElementById('stat-total');
  dom.statFlagged    = document.getElementById('stat-flagged');
  dom.statSafe       = document.getElementById('stat-safe');
  dom.statThreshold  = document.getElementById('stat-threshold');

  dom.filterBar      = document.getElementById('filter-bar');
  dom.filterBtns     = document.querySelectorAll('.filter-btn');
  dom.clearBtn       = document.getElementById('clear-btn');

  dom.resultsSection = document.getElementById('results-section');
  dom.resultsGrid    = document.getElementById('results-grid');
  dom.noResultsMsg   = document.getElementById('no-results-msg');

  dom.modal          = document.getElementById('suggestion-modal');
  dom.modalBackdrop  = document.getElementById('modal-backdrop');
  dom.modalCloseBtn  = document.getElementById('modal-close-btn');
  dom.modalTitle     = document.getElementById('modal-title');
  dom.modalQuery     = document.getElementById('modal-query');
  dom.modalLoading   = document.getElementById('modal-loading');
  dom.modalError     = document.getElementById('modal-error');
  dom.suggestionsCarousel = document.getElementById('suggestions-carousel');

  dom.cardTemplate        = document.getElementById('image-card-tpl');
  dom.suggestionCardTpl   = document.getElementById('suggestion-card-tpl');

  bindEvents();
});

// ============================================================
// Event binding
// ============================================================

function bindEvents() {
  dom.scanForm.addEventListener('submit', handleScanSubmit);
  dom.clearBtn.addEventListener('click', handleClear);

  dom.filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      currentFilter = btn.dataset.filter;
      dom.filterBtns.forEach(b => b.classList.toggle('filter-btn--active', b === btn));
      renderFilteredResults();
    });
  });

  dom.modalCloseBtn.addEventListener('click', closeModal);
  dom.modalBackdrop.addEventListener('click', closeModal);

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !dom.modal.hidden) {
      closeModal();
    }
  });
}

// ============================================================
// Scan form submission
// ============================================================

async function handleScanSubmit(e) {
  e.preventDefault();
  hideError();

  const target = dom.targetInput.value.trim();
  if (!target) {
    showError('Please enter a directory path or website URL.');
    dom.targetInput.focus();
    return;
  }

  const rawThreshold = parseFloat(dom.thresholdInput.value);
  if (isNaN(rawThreshold) || rawThreshold < 0 || rawThreshold > 1) {
    showError('Threshold must be a number between 0.0 and 1.0.');
    dom.thresholdInput.focus();
    return;
  }

  setScanningState(true);
  showProgress('Scanning images\u2026');
  hideResults();

  try {
    const data = await postScan(target, rawThreshold);
    allReports = data.images || [];
    renderSummary(data);
    renderFilteredResults();
    showResults();
  } catch (err) {
    showError(err.message || 'An unexpected error occurred. Please try again.');
    hideProgress();
  } finally {
    setScanningState(false);
    hideProgress();
  }
}

// ============================================================
// API calls
// ============================================================

/**
 * POST /api/scan
 * @param {string} target
 * @param {number} threshold
 * @returns {Promise<Object>}
 */
async function postScan(target, threshold) {
  const resp = await fetch('/api/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target, threshold }),
  });

  const data = await resp.json();

  if (!resp.ok) {
    const msg = data.message || data.error || `Server returned ${resp.status}`;
    throw new Error(msg);
  }

  return data;
}

/**
 * GET /api/suggestions?q=...&per_page=...
 * @param {string} query
 * @param {number} perPage
 * @returns {Promise<Object>}
 */
async function fetchSuggestions(query, perPage = 5) {
  const params = new URLSearchParams({ q: query, per_page: String(perPage) });
  const resp = await fetch(`/api/suggestions?${params}`);
  const data = await resp.json();

  if (!resp.ok) {
    const msg = data.message || data.error || `Server returned ${resp.status}`;
    throw new Error(msg);
  }

  return data;
}

// ============================================================
// Rendering: summary bar
// ============================================================

function renderSummary(data) {
  const total   = data.total || 0;
  const flagged = data.flagged_count || 0;
  const safe    = total - flagged;

  dom.statTotal.textContent     = total;
  dom.statFlagged.textContent   = flagged;
  dom.statSafe.textContent      = safe;
  dom.statThreshold.textContent = (typeof data.threshold === 'number')
    ? data.threshold.toFixed(2)
    : '–';

  dom.summaryBar.hidden = false;
  dom.filterBar.hidden  = false;
}

// ============================================================
// Rendering: results grid
// ============================================================

function renderFilteredResults() {
  dom.resultsGrid.innerHTML = '';

  const filtered = allReports.filter(report => {
    if (currentFilter === 'flagged') return report.ai_is_flagged === true;
    if (currentFilter === 'safe')    return report.ai_is_flagged === false;
    return true;
  });

  if (filtered.length === 0) {
    dom.noResultsMsg.hidden = false;
  } else {
    dom.noResultsMsg.hidden = true;
    filtered.forEach(report => {
      const card = buildImageCard(report);
      dom.resultsGrid.appendChild(card);
    });
  }

  dom.resultsSection.hidden = false;
}

// ============================================================
// Image card construction
// ============================================================

/**
 * Build a DOM node for a single image report.
 * @param {Object} report
 * @returns {HTMLElement}
 */
function buildImageCard(report) {
  const tpl   = dom.cardTemplate.content.cloneNode(true);
  const card  = tpl.querySelector('.image-card');

  card.dataset.source      = report.source || '';
  card.dataset.flagged     = String(report.ai_is_flagged === true);
  card.dataset.hasError    = String(!!report.error);

  // ---- Image ----
  const imgEl        = card.querySelector('.card-img');
  const placeholder  = card.querySelector('.card-img-placeholder');
  const sourceLabel  = report.source || '';
  const isRemote     = report.is_remote === true;

  if (isRemote) {
    imgEl.src = report.source;
    imgEl.alt = report.alt_text || 'Remote image';
  } else {
    // Local images cannot be served directly; show placeholder.
    imgEl.style.display = 'none';
    placeholder.classList.add('card-img-placeholder--visible');
  }

  imgEl.addEventListener('error', () => {
    imgEl.style.display = 'none';
    placeholder.classList.add('card-img-placeholder--visible');
  });

  imgEl.addEventListener('load', () => {
    placeholder.style.display = 'none';
  });

  // ---- Flag badge ----
  const badgeAI    = card.querySelector('.flag-badge--ai');
  const badgeSafe  = card.querySelector('.flag-badge--safe');
  const badgeError = card.querySelector('.flag-badge--error');

  if (report.error) {
    badgeAI.hidden   = true;
    badgeSafe.hidden = true;
    badgeError.hidden = false;
    card.classList.add('image-card--error');
  } else if (report.ai_is_flagged === true) {
    badgeAI.hidden    = false;
    badgeSafe.hidden  = true;
    badgeError.hidden = true;
    card.classList.add('image-card--flagged');
  } else {
    badgeAI.hidden    = true;
    badgeSafe.hidden  = false;
    badgeError.hidden = true;
    card.classList.add('image-card--safe');
  }

  // ---- Source label ----
  const sourceEl = card.querySelector('.card-source');
  const shortSource = truncateSource(sourceLabel);
  sourceEl.textContent = shortSource;
  sourceEl.title       = sourceLabel;

  // ---- Score bar ----
  const scoreRow   = card.querySelector('.score-row');
  const scoreBar   = card.querySelector('.score-bar');
  const scoreValue = card.querySelector('.score-value');

  if (typeof report.ai_score === 'number') {
    const pct = Math.round(report.ai_score * 100);
    scoreBar.style.width     = pct + '%';
    scoreBar.style.setProperty('--score', report.ai_score);
    scoreValue.textContent   = pct + '%';
    scoreBar.classList.toggle('score-bar--high', report.ai_score >= 0.7);
    scoreBar.classList.toggle('score-bar--mid',  report.ai_score >= 0.4 && report.ai_score < 0.7);
    scoreBar.classList.toggle('score-bar--low',  report.ai_score < 0.4);
  } else {
    scoreRow.hidden = true;
  }

  // ---- Suggest button ----
  const suggestBtn = card.querySelector('.suggest-btn');
  if (report.error || report.ai_is_flagged === false) {
    // Still show suggest button for all images, but style differently for safe
    if (!isRemote && !report.error) {
      suggestBtn.classList.add('btn--ghost');
      suggestBtn.classList.remove('btn--suggest');
    }
  }
  suggestBtn.addEventListener('click', () => {
    openSuggestionsModal(report);
  });

  // ---- Toggle metadata ----
  const toggleBtn = card.querySelector('.toggle-meta-btn');
  const metaPanel = card.querySelector('.meta-panel');

  toggleBtn.addEventListener('click', () => {
    const expanded = toggleBtn.getAttribute('aria-expanded') === 'true';
    toggleBtn.setAttribute('aria-expanded', String(!expanded));
    metaPanel.hidden = expanded;
    toggleBtn.querySelector('.chevron-icon').classList.toggle('chevron-icon--up', !expanded);
  });

  // Populate metadata
  populateMetaPanel(metaPanel, report);

  return card;
}

/**
 * Populate the expandable metadata panel.
 * @param {HTMLElement} panel
 * @param {Object} report
 */
function populateMetaPanel(panel, report) {
  const dl = panel.querySelector('.meta-list');
  dl.innerHTML = '';

  const entries = [
    ['Format',      report.format],
    ['Mode',        report.mode],
    ['Dimensions',  (report.width && report.height)
                      ? `${report.width} × ${report.height} px`
                      : null],
    ['Aspect ratio', report.aspect_ratio != null ? report.aspect_ratio.toFixed(3) : null],
    ['File size',   report.file_size_bytes != null
                      ? formatBytes(report.file_size_bytes)
                      : null],
    ['AI score',    typeof report.ai_score === 'number'
                      ? `${(report.ai_score * 100).toFixed(1)}%`
                      : null],
    ['Verdict',     report.ai_verdict],
    ['Model mode',  report.ai_model_mode],
    ['Threshold',   typeof report.ai_threshold === 'number'
                      ? report.ai_threshold.toFixed(2)
                      : null],
    ['Alt text',    report.alt_text || null],
    ['Error',       report.error || null],
  ];

  entries.forEach(([label, value]) => {
    if (value == null) return;
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = String(value);
    if (label === 'Error') dd.classList.add('meta-error');
    dl.appendChild(dt);
    dl.appendChild(dd);
  });

  // Colour palette swatches
  const paletteRow = panel.querySelector('.palette-row');
  paletteRow.innerHTML = '';
  const colours = report.colour_stats && report.colour_stats.dominant_colours;
  if (colours && colours.length > 0) {
    const label = document.createElement('span');
    label.className = 'palette-label';
    label.textContent = 'Palette:';
    paletteRow.appendChild(label);

    colours.forEach(rgb => {
      const swatch = document.createElement('span');
      swatch.className  = 'colour-swatch';
      const cssColour   = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      swatch.style.background = cssColour;
      swatch.title = cssColour;
      paletteRow.appendChild(swatch);
    });
  }
}

// ============================================================
// Suggestions modal
// ============================================================

/**
 * Open the suggestions modal and fetch alternatives for a given image report.
 * @param {Object} report
 */
async function openSuggestionsModal(report) {
  activeSuggestionSource = report.source;
  dom.suggestionsCarousel.innerHTML = '';
  dom.modalError.hidden    = true;
  dom.modalLoading.hidden  = true;
  dom.modalError.textContent = '';

  const query = buildSuggestionQuery(report);
  dom.modalQuery.textContent = query ? `Query: "${query}"` : '';

  dom.modal.hidden = false;
  document.body.classList.add('modal-open');
  dom.modalCloseBtn.focus();

  if (!query) {
    showModalError('Could not derive a search query for this image.');
    return;
  }

  dom.modalLoading.hidden = false;

  try {
    const data = await fetchSuggestions(query, 6);
    dom.modalLoading.hidden = true;

    if (!data.suggestions || data.suggestions.length === 0) {
      showModalError('No royalty-free alternatives found for this query. Try a different image.');
      return;
    }

    renderSuggestionsCarousel(data.suggestions);
  } catch (err) {
    dom.modalLoading.hidden = true;
    showModalError(`Failed to fetch suggestions: ${err.message}`);
  }
}

/**
 * Close the suggestions modal.
 */
function closeModal() {
  dom.modal.hidden = true;
  document.body.classList.remove('modal-open');
  activeSuggestionSource = null;
}

/**
 * Render suggestion cards inside the carousel container.
 * @param {Array<Object>} suggestions
 */
function renderSuggestionsCarousel(suggestions) {
  dom.suggestionsCarousel.innerHTML = '';

  suggestions.forEach(s => {
    const tpl  = dom.suggestionCardTpl.content.cloneNode(true);
    const card = tpl.querySelector('.suggestion-card');

    const link = card.querySelector('.suggestion-link');
    link.href  = s.foreign_landing_url || s.url;

    const img  = card.querySelector('.suggestion-img');
    img.src    = s.thumbnail || s.url;
    img.alt    = s.title || 'Royalty-free image';

    img.addEventListener('error', () => {
      img.src = s.url; // fallback to full URL
    });

    const titleEl   = card.querySelector('.suggestion-title');
    titleEl.textContent = s.title || 'Untitled';

    const creatorEl = card.querySelector('.suggestion-creator');
    if (s.creator) {
      creatorEl.textContent = `by ${s.creator}`;
    } else {
      creatorEl.hidden = true;
    }

    const licenceEl = card.querySelector('.suggestion-licence');
    licenceEl.textContent = formatLicence(s.licence, s.licence_version);
    licenceEl.href        = s.licence_url || '#';
    if (!s.licence_url) licenceEl.removeAttribute('href');

    dom.suggestionsCarousel.appendChild(card);
  });
}

/**
 * Show an error message inside the modal.
 * @param {string} message
 */
function showModalError(message) {
  dom.modalError.textContent = message;
  dom.modalError.hidden      = false;
}

// ============================================================
// Query derivation
// ============================================================

/**
 * Build a search query string from an image report.
 * Priority: alt_text > dominant colour adjective + format fallback.
 * @param {Object} report
 * @returns {string}
 */
function buildSuggestionQuery(report) {
  // 1. Use alt text if non-trivial
  if (report.alt_text && report.alt_text.trim().length > 3) {
    return report.alt_text.trim().slice(0, 100);
  }

  // 2. Derive from filename / URL path
  const source = report.source || '';
  const parts  = source.replace(/\?.*$/, '').split(/[\/\\]/);
  const filename = parts[parts.length - 1] || '';
  const stem = filename.replace(/\.[^.]+$/, '').replace(/[-_]+/g, ' ').trim();

  if (stem.length > 3 && stem.length < 80) {
    return stem;
  }

  // 3. Fallback: generic photography query
  return 'royalty free photography';
}

// ============================================================
// UI state helpers
// ============================================================

function setScanningState(scanning) {
  dom.scanBtn.disabled = scanning;
  dom.targetInput.disabled = scanning;
  dom.thresholdInput.disabled = scanning;
  dom.scanBtn.classList.toggle('btn--loading', scanning);
  if (scanning) {
    dom.scanBtn.setAttribute('aria-busy', 'true');
  } else {
    dom.scanBtn.removeAttribute('aria-busy');
  }
}

function showProgress(message) {
  dom.progressMsg.textContent = message;
  dom.progressPanel.hidden = false;
}

function hideProgress() {
  dom.progressPanel.hidden = true;
}

function showResults() {
  dom.resultsSection.hidden = false;
}

function hideResults() {
  dom.resultsSection.hidden = true;
  dom.summaryBar.hidden     = true;
  dom.filterBar.hidden      = true;
  dom.resultsGrid.innerHTML = '';
}

function showError(message) {
  dom.formError.textContent = message;
  dom.formError.hidden      = false;
}

function hideError() {
  dom.formError.textContent = '';
  dom.formError.hidden      = true;
}

function handleClear() {
  allReports    = [];
  currentFilter = 'all';
  dom.filterBtns.forEach(b => {
    b.classList.toggle('filter-btn--active', b.dataset.filter === 'all');
  });
  hideResults();
  hideError();
  dom.targetInput.value    = '';
  dom.thresholdInput.value = '0.5';
  dom.targetInput.focus();
}

// ============================================================
// Formatting utilities
// ============================================================

/**
 * Format a byte count as a human-readable string.
 * @param {number} bytes
 * @returns {string}
 */
function formatBytes(bytes) {
  if (bytes < 1024)       return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

/**
 * Truncate a long source path/URL for display.
 * @param {string} source
 * @returns {string}
 */
function truncateSource(source) {
  if (!source) return '(unknown)';
  if (source.length <= 55) return source;
  const start = source.slice(0, 20);
  const end   = source.slice(-28);
  return `${start}\u2026${end}`;
}

/**
 * Format a Creative Commons licence for display.
 * @param {string} licence
 * @param {string|null} version
 * @returns {string}
 */
function formatLicence(licence, version) {
  if (!licence) return 'Unknown licence';
  const upper = licence.toUpperCase();
  if (upper === 'CC0') return 'CC0 (Public Domain)';
  const ver = version ? ` ${version}` : '';
  return `CC ${upper}${ver}`;
}
