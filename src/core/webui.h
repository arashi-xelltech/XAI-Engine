// src/core/webui.h
#pragma once

namespace xai {
namespace webui {

inline const char* get_chat_html() {
    return R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XAI Engine</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-primary:    #0a0a0a;
    --bg-secondary:  #111111;
    --bg-tertiary:   #181818;
    --bg-hover:      #222222;
    --border:        #2a2a2a;
    --border-hover:  #3d3d3d;
    --border-focus:  #555555;
    --text-primary:  #e8e8e8;
    --text-secondary:#999999;
    --text-muted:    #666666;
    --text-dim:      #4a4a4a;
    --accent:        #ffffff;
    --accent-hover:  #dddddd;
    --radius-sm:     10px;
    --radius-md:     14px;
    --radius-lg:     18px;
    --radius-full:   9999px;
    --font: 'PT Mono', 'Courier New', 'Consolas', 'Liberation Mono', monospace;
    --tr-fast: 0.15s ease;
    --tr:      0.25s ease;
    --tr-slow: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  }

  body {
    font-family: var(--font);
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    display: flex;
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* ── SIDEBAR ───────────────────────────────────────── */
  .sidebar {
    width: 300px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    transition: width var(--tr-slow), opacity var(--tr), border-color var(--tr);
    overflow: hidden;
    will-change: width;
  }

  .sidebar.closed {
    width: 0;
    opacity: 0;
    border-right-color: transparent;
  }

  .sidebar-inner {
    width: 300px;
    min-width: 300px;
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .sidebar-header {
    height: 48px;
    padding: 0 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    flex-shrink: 0;
  }

  .sidebar-header h2 {
    font-family: var(--font);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text-secondary);
  }

  .sidebar-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 12px 14px 20px;
    min-height: 0;
  }

  .sidebar-scroll::-webkit-scrollbar { width: 3px; }
  .sidebar-scroll::-webkit-scrollbar-track { background: transparent; }
  .sidebar-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* ── СЕКЦИИ ПАРАМЕТРОВ ─────────────────────────────── */
  .param-section {
    margin-bottom: 5px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .section-toggle-btn {
    width: 100%;
    background: var(--bg-tertiary);
    border: none;
    padding: 9px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    color: var(--text-muted);
    font-family: var(--font);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    user-select: none;
    transition: background var(--tr-fast);
  }

  .section-toggle-btn:hover { background: var(--bg-hover); }

  .section-chevron {
    width: 12px;
    height: 12px;
    flex-shrink: 0;
    transition: transform var(--tr);
    stroke: var(--text-muted);
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
  }

  .section-toggle-btn.open .section-chevron {
    transform: rotate(90deg);
  }

  .section-collapsible {
    max-height: 0;
    overflow: hidden;
    transition: max-height var(--tr-slow), opacity var(--tr);
    opacity: 0;
  }

  .section-collapsible.open {
    max-height: 600px;
    opacity: 1;
  }

  .section-body {
    padding: 2px 0;
    border-top: 1px solid var(--border);
  }

  /* ── СТРОКА ПАРАМЕТРА ──────────────────────────────── */
  .param-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 12px;
    gap: 10px;
  }

  .param-row + .param-row {
    border-top: 1px solid var(--border);
  }

  .param-label {
    font-size: 11px;
    color: var(--text-secondary);
    user-select: none;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .param-row input[type="number"] {
    width: 68px;
    padding: 5px 7px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-full);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
    text-align: center;
    outline: none;
    transition: border-color var(--tr-fast);
    -moz-appearance: textfield;
    flex-shrink: 0;
  }

  .param-row input[type="number"]::-webkit-inner-spin-button,
  .param-row input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; }

  .param-row input[type="number"]:focus { border-color: var(--border-focus); }

  .toggle {
    -webkit-appearance: none;
    appearance: none;
    width: 34px;
    height: 18px;
    background: var(--border);
    border-radius: 9px;
    cursor: pointer;
    position: relative;
    transition: background var(--tr-fast);
    flex-shrink: 0;
    border: none;
    outline: none;
  }

  .toggle:checked { background: var(--text-secondary); }

  .toggle::after {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    background: var(--text-primary);
    border-radius: 50%;
    transition: transform var(--tr-fast);
  }

  .toggle:checked::after { transform: translateX(16px); }

  .seed-field-row {
    padding: 0 12px 8px;
    display: none;
    border-top: 1px solid var(--border);
  }

  .seed-field-row.visible { display: block; }

  .seed-field-row input {
    width: 100%;
    padding: 6px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-full);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
    text-align: center;
    outline: none;
    transition: border-color var(--tr-fast);
    -moz-appearance: textfield;
    margin-top: 8px;
  }

  .seed-field-row input::-webkit-inner-spin-button,
  .seed-field-row input::-webkit-outer-spin-button { -webkit-appearance: none; }

  .seed-field-row input:focus { border-color: var(--border-focus); }

  /* ── PROMPT TEMPLATE ───────────────────────────────── */
  .template-section {
    margin-bottom: 5px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .template-toggle-btn {
    width: 100%;
    background: var(--bg-tertiary);
    border: none;
    padding: 9px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    color: var(--text-muted);
    font-family: var(--font);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    user-select: none;
    transition: background var(--tr-fast);
  }

  .template-toggle-btn:hover { background: var(--bg-hover); }

  .chevron {
    width: 12px;
    height: 12px;
    flex-shrink: 0;
    transition: transform var(--tr);
    stroke: var(--text-muted);
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
  }

  .template-toggle-btn.open .chevron { transform: rotate(90deg); }

  .template-collapsible {
    max-height: 0;
    overflow: hidden;
    transition: max-height var(--tr-slow), opacity var(--tr);
    opacity: 0;
  }

  .template-collapsible.open {
    max-height: 900px;
    opacity: 1;
  }

  .template-body {
    padding: 10px 12px;
    border-top: 1px solid var(--border);
  }

  .template-textarea {
    width: 100%;
    min-height: 110px;
    padding: 9px 11px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
    line-height: 1.65;
    resize: vertical;
    outline: none;
    transition: border-color var(--tr-fast);
    tab-size: 2;
  }

  .template-textarea:focus { border-color: var(--border-focus); }

  .template-hint {
    margin-top: 7px;
    font-size: 10px;
    color: var(--text-dim);
    line-height: 1.5;
  }

  /* ── КАСТОМНЫЕ ПЕРЕМЕННЫЕ — в сворачиваемой секции ── */
  .vars-section {
    margin-bottom: 5px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    display: none;
  }

  .vars-section.has-vars { display: block; }

  .vars-toggle-btn {
    width: 100%;
    background: var(--bg-tertiary);
    border: none;
    padding: 9px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    color: var(--text-muted);
    font-family: var(--font);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    user-select: none;
    transition: background var(--tr-fast);
  }

  .vars-toggle-btn:hover { background: var(--bg-hover); }

  .vars-chevron {
    width: 12px;
    height: 12px;
    flex-shrink: 0;
    transition: transform var(--tr);
    stroke: var(--text-muted);
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
  }

  .vars-toggle-btn.open .vars-chevron { transform: rotate(90deg); }

  .vars-collapsible {
    max-height: 0;
    overflow: hidden;
    transition: max-height var(--tr-slow), opacity var(--tr);
    opacity: 0;
  }

  .vars-collapsible.open {
    max-height: 800px;
    opacity: 1;
  }

  .vars-body {
    padding: 10px 12px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .var-row { display: flex; flex-direction: column; gap: 4px; }

  .var-row label {
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding-left: 2px;
  }

  .var-row input {
    width: 100%;
    padding: 6px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-full);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
    outline: none;
    transition: border-color var(--tr-fast);
  }

  .var-row input:focus { border-color: var(--border-focus); }

  /* ── MAIN ──────────────────────────────────────────── */
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: var(--bg-primary);
  }

  .top-bar {
    height: 48px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .icon-btn {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background var(--tr-fast), border-color var(--tr-fast);
    padding: 0;
  }

  .icon-btn:hover {
    background: var(--bg-hover);
    border-color: var(--border-hover);
  }

  .icon-btn svg {
    width: 14px;
    height: 14px;
    stroke: var(--text-secondary);
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
    transition: stroke var(--tr-fast);
    display: block;
  }

  .icon-btn:hover svg { stroke: var(--text-primary); }

  .top-bar-title {
    font-family: var(--font);
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .top-bar-spacer { flex: 1; }

  /* ── CHAT ──────────────────────────────────────────── */
  .chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    min-height: 0;
    /* GPU-слой для плавного скролла */
    will-change: scroll-position;
    -webkit-overflow-scrolling: touch;
  }

  .chat-container::-webkit-scrollbar { width: 3px; }
  .chat-container::-webkit-scrollbar-track { background: transparent; }
  .chat-container::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .welcome-screen {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px 20px;
    animation: fadeUp 0.6s ease;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .welcome-screen h1 {
    font-family: var(--font);
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
  }

  .welcome-screen p {
    font-size: 12px;
    color: var(--text-muted);
    max-width: 380px;
    line-height: 1.7;
  }

  .welcome-screen .hint {
    margin-top: 20px;
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.5px;
  }

  /* ── СООБЩЕНИЯ ─────────────────────────────────────── */
  .message-row {
    display: flex;
    flex-direction: column;
    padding: 0 28px;
    /* анимация появления */
    animation: msgIn 0.28s cubic-bezier(0.4, 0, 0.2, 1) both;
    transform-origin: bottom center;
  }

  @keyframes msgIn {
    from { opacity: 0; transform: translateY(8px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0)  scale(1); }
  }

  .message-row.user { align-items: flex-end; }
  .message-row.ai   { align-items: flex-start; }

  .message-label {
    font-family: var(--font);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-dim);
    margin-bottom: 5px;
    padding: 0 4px;
  }

  .message-bubble {
    max-width: 75%;
    padding: 12px 16px;
    font-family: var(--font);
    font-size: 13px;
    line-height: 1.75;
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap; /* ← ПЕРЕНОС СТРОК */
  }

  .message-row.user .message-bubble {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
    color: var(--text-primary);
    white-space: pre-wrap;
  }

  .message-row.ai .message-bubble {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 4px var(--radius-lg) var(--radius-lg) var(--radius-lg);
    color: var(--text-primary);
    max-width: 85%;
    /* белое пространство управляется явно через renderMarkdown */
    white-space: normal;
  }

  /* ── CHAPTER DIVIDER ───────────────────────────────── */
  .chapter-divider {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 14px 0 10px;
    width: 100%;
  }

  .chapter-divider-line {
    flex: 1;
    height: 1px;
    background: var(--border-hover);
    border-radius: 1px;
  }

  .chapter-divider-label {
    font-size: 9px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    white-space: nowrap;
    padding: 0 2px;
    user-select: none;
  }

  /* ── CODE BLOCKS ───────────────────────────────────── */
  .message-bubble pre {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 11px 13px;
    overflow-x: auto;
    font-family: var(--font);
    font-size: 11px;
    line-height: 1.5;
    margin: 8px 0;
    white-space: pre;
  }

  .message-bubble code {
    font-family: var(--font);
    font-size: 11px;
    background: var(--bg-tertiary);
    padding: 2px 5px;
    border-radius: 4px;
    border: 1px solid var(--border);
    white-space: pre-wrap;
  }

  .message-bubble pre code {
    background: none;
    border: none;
    padding: 0;
    white-space: pre;
  }

  .message-bubble strong { font-weight: 700; color: #fff; }
  .message-bubble em { font-style: italic; color: var(--text-secondary); }

  /* ── CURSOR ────────────────────────────────────────── */
  .cursor-blink {
    display: inline-block;
    width: 2px;
    height: 13px;
    background: var(--text-primary);
    margin-left: 1px;
    vertical-align: text-bottom;
    border-radius: 1px;
    animation: cursorPulse 0.7s infinite;
    will-change: opacity;
  }

  @keyframes cursorPulse {
    0%, 48% { opacity: 1; }
    50%, 100% { opacity: 0; }
  }

  /* ── STREAMING TOKEN FADE-IN ───────────────────────── */
  @keyframes tokenFade {
    from { opacity: 0; }
    to   { opacity: 1; }
  }

  /* span-обёртки для плавного появления текстовых чанков */
  .token-chunk {
    animation: tokenFade 0.18s ease both;
  }

  .stopped-indicator {
    font-size: 10px;
    color: var(--text-dim);
    font-style: italic;
    margin-top: 5px;
    padding-left: 4px;
  }

  /* ── INPUT BAR ─────────────────────────────────────── */
  .input-bar {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .input-textarea {
    flex: 1;
    padding: 10px 18px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-full);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 13px;
    line-height: 1.5;
    resize: none;
    outline: none;
    max-height: 140px;
    transition: border-color var(--tr-fast);
  }

  .input-textarea:focus { border-color: var(--border-focus); }
  .input-textarea::placeholder { color: var(--text-dim); }

  .send-stop-btn {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: var(--text-primary);
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background var(--tr-fast), transform 0.1s ease;
    position: relative;
  }

  .send-stop-btn:hover { background: var(--accent-hover); transform: scale(1.05); }
  .send-stop-btn:active { transform: scale(0.95); }
  .send-stop-btn:disabled { opacity: 0.25; cursor: not-allowed; transform: none; }

  .send-stop-btn svg {
    position: absolute;
    transition: opacity var(--tr-fast), transform var(--tr-fast);
    display: block;
  }

  .send-stop-btn .icon-send {
    width: 16px; height: 16px;
    fill: var(--bg-primary);
    stroke: none;
    opacity: 1;
    transform: scale(1);
  }

  .send-stop-btn .icon-stop {
    width: 13px; height: 13px;
    fill: var(--bg-primary);
    stroke: none;
    opacity: 0;
    transform: scale(0.5);
  }

  .send-stop-btn.generating .icon-send { opacity: 0; transform: scale(0.5); }
  .send-stop-btn.generating .icon-stop { opacity: 1; transform: scale(1); }

  /* ── RESPONSIVE ────────────────────────────────────── */
  @media (max-width: 768px) {
    .sidebar {
      position: fixed;
      left: 0; top: 0; bottom: 0;
      z-index: 100;
      box-shadow: 8px 0 30px rgba(0,0,0,0.5);
    }
    .sidebar.closed { width: 0; }
    .message-row { padding: 0 4px; }
    .message-bubble { max-width: 92% !important; }
    .chat-container { padding: 14px 8px; }
    .input-bar { padding: 8px 10px; }
  }
</style>
</head>
<body>

<!-- ── SIDEBAR ───────────────────────────────────────────── -->
<div class="sidebar closed" id="sidebar">
  <div class="sidebar-inner">

    <div class="sidebar-header">
      <h2>Parameters</h2>
    </div>

    <div class="sidebar-scroll">

      <!-- Sampling — сворачиваемая секция -->
      <div class="param-section">
        <button class="section-toggle-btn open" id="sectionSamplingBtn" onclick="toggleSection('sampling')">
          <span>Sampling</span>
          <svg class="section-chevron" viewBox="0 0 16 16"><polyline points="5,2 11,8 5,14"/></svg>
        </button>
        <div class="section-collapsible open" id="sectionSamplingBody">
          <div class="section-body">
            <div class="param-row">
              <span class="param-label">Temperature</span>
              <input type="number" id="temperature" value="0.8" step="0.05" min="0" max="5">
            </div>
            <div class="param-row">
              <span class="param-label">Top-K</span>
              <input type="number" id="topK" value="50" min="1" max="1000">
            </div>
            <div class="param-row">
              <span class="param-label">Top-P</span>
              <input type="number" id="topP" value="0.9" step="0.02" min="0" max="1">
            </div>
            <div class="param-row">
              <span class="param-label">Rep. Penalty</span>
              <input type="number" id="repPenalty" value="1.1" step="0.05" min="1" max="3">
            </div>
          </div>
        </div>
      </div>

      <!-- Generation — сворачиваемая секция -->
      <div class="param-section">
        <button class="section-toggle-btn open" id="sectionGenerationBtn" onclick="toggleSection('generation')">
          <span>Generation</span>
          <svg class="section-chevron" viewBox="0 0 16 16"><polyline points="5,2 11,8 5,14"/></svg>
        </button>
        <div class="section-collapsible open" id="sectionGenerationBody">
          <div class="section-body">
            <div class="param-row">
              <span class="param-label">Max Tokens</span>
              <input type="number" id="maxTokens" value="512" min="1" max="8192">
            </div>
            <div class="param-row">
              <span class="param-label">Manual Seed</span>
              <input type="checkbox" class="toggle" id="manualSeedToggle" onchange="onManualSeedToggle()">
            </div>
            <div class="seed-field-row" id="seedFieldRow">
              <input type="number" id="seed" value="42" min="0" placeholder="Seed value">
            </div>
            <div class="param-row">
              <span class="param-label">Ignore EOS</span>
              <input type="checkbox" class="toggle" id="ignoreEOS">
            </div>
            <div class="param-row">
              <span class="param-label">Streaming</span>
              <input type="checkbox" class="toggle" id="streaming" checked>
            </div>
          </div>
        </div>
      </div>

      <!-- Prompt Template — сворачиваемая -->
      <div class="template-section">
        <button class="template-toggle-btn" id="templateToggleBtn" onclick="toggleTemplate()">
          <span>Prompt Template</span>
          <svg class="chevron" viewBox="0 0 16 16"><polyline points="5,2 11,8 5,14"/></svg>
        </button>
        <div class="template-collapsible" id="templateCollapsible">
          <div class="template-body">
            <textarea
              class="template-textarea"
              id="templateEditor"
              oninput="onTemplateChange()"
              placeholder="Use |variable name| for variables.&#10;|USH| = user input, |MSH| = where model starts generating.&#10;&#10;Example:&#10;System: You are helpful.&#10;User: |USH|&#10;Assistant: |MSH|"
            >User: |USH|
Assistant: |MSH|</textarea>
            <div class="template-hint">
              |USH| — user input &nbsp;·&nbsp; |MSH| — generation start &nbsp;·&nbsp; |name| — variable
            </div>
          </div>
        </div>
      </div>

      <!-- Custom Variables — сворачиваемая, скрыта если нет переменных -->
      <div class="vars-section" id="varsSection">
        <button class="vars-toggle-btn open" id="varsToggleBtn" onclick="toggleVars()">
          <span>Variables</span>
          <svg class="vars-chevron" viewBox="0 0 16 16"><polyline points="5,2 11,8 5,14"/></svg>
        </button>
        <div class="vars-collapsible open" id="varsCollapsible">
          <div class="vars-body" id="varsBody"></div>
        </div>
      </div>

    </div><!-- /sidebar-scroll -->
  </div><!-- /sidebar-inner -->
</div><!-- /sidebar -->

<!-- ── MAIN ─────────────────────────────────────────────── -->
<div class="main">

  <div class="top-bar">
    <button class="icon-btn" onclick="toggleSidebar()" title="Toggle settings">
      <svg viewBox="0 0 24 24">
        <line x1="4" y1="7"  x2="20" y2="7"/>
        <line x1="4" y1="12" x2="20" y2="12"/>
        <line x1="4" y1="17" x2="20" y2="17"/>
      </svg>
    </button>
    <span class="top-bar-title">XAI Engine</span>
    <span class="top-bar-spacer"></span>
    <button class="icon-btn" onclick="clearChat()" title="Clear conversation">
      <svg viewBox="0 0 24 24">
        <polyline points="3 6 5 6 21 6"/>
        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
        <path d="M10 11v6M14 11v6"/>
        <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
      </svg>
    </button>
  </div>

  <div class="chat-container" id="chatContainer">
    <div class="welcome-screen" id="welcomeScreen">
      <h1>XAI Engine</h1>
      <p>High-performance local inference.<br>Open settings to configure parameters and prompt templates.</p>
      <span class="hint">Enter — send &nbsp;·&nbsp; Shift+Enter — new line</span>
    </div>
  </div>

  <div class="input-bar">
    <textarea
      class="input-textarea"
      id="userInput"
      rows="1"
      placeholder="Type a message..."
      onkeydown="handleInputKey(event)"
      oninput="autoResizeInput()"
    ></textarea>
    <button class="send-stop-btn" id="sendStopBtn" onclick="handleSendStop()" title="Send">
      <svg class="icon-send" viewBox="0 0 24 24">
        <path d="M2 21l21-9L2 3v7l15 2-15 2z"/>
      </svg>
      <svg class="icon-stop" viewBox="0 0 24 24">
        <rect x="4" y="4" width="16" height="16" rx="2.5"/>
      </svg>
    </button>
  </div>

</div>

<!-- ── SCRIPT ─────────────────────────────────────────────── -->
<script>
(function () {
  'use strict';

  /* ── DOM refs ── */
  const sidebar           = document.getElementById('sidebar');
  const chatContainer     = document.getElementById('chatContainer');
  const userInput         = document.getElementById('userInput');
  const sendStopBtn       = document.getElementById('sendStopBtn');
  const templateEditor    = document.getElementById('templateEditor');
  const seedFieldRow      = document.getElementById('seedFieldRow');
  const manualSeedToggle  = document.getElementById('manualSeedToggle');
  const varsSection       = document.getElementById('varsSection');
  const varsBody          = document.getElementById('varsBody');

  /* ── State ── */
  let isGenerating      = false;
  let abortController   = null;
  let currentRequestId  = null;
  let templateVariables = {};

  /* ── Scroll throttle ── */
  let scrollScheduled   = false;

  function scheduleScroll() {
    if (scrollScheduled) return;
    scrollScheduled = true;
    requestAnimationFrame(() => {
      chatContainer.scrollTop = chatContainer.scrollHeight;
      scrollScheduled = false;
    });
  }

  /* ════════════════════════════════════════
     SIDEBAR
  ════════════════════════════════════════ */
  window.toggleSidebar = function () {
    sidebar.classList.toggle('closed');
  };

  /* ── Section collapse (Sampling / Generation) ── */
  window.toggleSection = function (name) {
    const btn  = document.getElementById('sectionSampling'  === 'sampling' ? 'sectionSamplingBtn'  : 'section' + capitalize(name) + 'Btn');
    const body = document.getElementById('sectionSampling'  === 'sampling' ? 'sectionSamplingBody' : 'section' + capitalize(name) + 'Body');
    // More robust lookup:
    const btnEl  = document.getElementById('section' + capitalize(name) + 'Btn');
    const bodyEl = document.getElementById('section' + capitalize(name) + 'Body');
    if (!btnEl || !bodyEl) return;
    btnEl.classList.toggle('open');
    bodyEl.classList.toggle('open');
  };

  function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  /* ── Template section ── */
  window.toggleTemplate = function () {
    const btn  = document.getElementById('templateToggleBtn');
    const body = document.getElementById('templateCollapsible');
    btn.classList.toggle('open');
    body.classList.toggle('open');
  };

  /* ── Variables section ── */
  window.toggleVars = function () {
    const btn  = document.getElementById('varsToggleBtn');
    const body = document.getElementById('varsCollapsible');
    btn.classList.toggle('open');
    body.classList.toggle('open');
  };

  /* ════════════════════════════════════════
     SEED
  ════════════════════════════════════════ */
  window.onManualSeedToggle = function () {
    seedFieldRow.classList.toggle('visible', manualSeedToggle.checked);
  };

  function getEffectiveSeed() {
    if (manualSeedToggle.checked) {
      return parseInt(document.getElementById('seed').value) || 0;
    }
    return Math.floor(Math.random() * 2147483647);
  }

  /* ════════════════════════════════════════
     TEMPLATE & VARIABLES
  ════════════════════════════════════════ */
  window.onTemplateChange = function () {
    parseTemplateVariables();
  };

  function parseTemplateVariables() {
    const text  = templateEditor.value;
    const regex = /\|([^|]+)\|/g;
    const found = [];
    let match;

    while ((match = regex.exec(text)) !== null) {
      const name = match[1].trim();
      if (name === 'USH' || name === 'MSH') continue;
      if (!found.includes(name)) found.push(name);
    }

    const next = {};
    found.forEach(n => {
      next[n] = (templateVariables[n] !== undefined) ? templateVariables[n] : '';
    });
    templateVariables = next;

    renderVarInputs(found);
  }

  function renderVarInputs(names) {
    if (names.length === 0) {
      varsSection.classList.remove('has-vars');
      varsBody.innerHTML = '';
      return;
    }

    varsSection.classList.add('has-vars');
    varsBody.innerHTML = names.map(name => {
      const val = escAttr(templateVariables[name] || '');
      return (
        '<div class="var-row">' +
          '<label>' + escHtml(name) + '</label>' +
          '<input type="text" value="' + val + '" ' +
            'data-var="' + escAttr(name) + '" ' +
            'placeholder="' + escAttr(name) + '" ' +
            'oninput="updateVar(this)">' +
        '</div>'
      );
    }).join('');
  }

  window.updateVar = function (el) {
    templateVariables[el.getAttribute('data-var')] = el.value;
  };

  function buildPrompt(userText) {
    let tmpl = templateEditor.value;

    tmpl = tmpl.replace(/\|([^|]+)\|/g, (match, raw) => {
      const name = raw.trim();
      if (name === 'MSH') return '|MSH|';
      if (name === 'USH') return userText;
      return (templateVariables[name] !== undefined) ? templateVariables[name] : '';
    });

    if (tmpl.indexOf('|MSH|') !== -1) {
      const parts = tmpl.split('|MSH|');
      return parts[0] + (parts[1] || '');
    }

    return tmpl + userText;
  }

  /* ════════════════════════════════════════
     INPUT
  ════════════════════════════════════════ */
  window.autoResizeInput = function () {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
  };

  window.handleInputKey = function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendStop();
    }
  };

  /* ════════════════════════════════════════
     SEND / STOP
  ════════════════════════════════════════ */
  window.handleSendStop = function () {
    if (isGenerating) stopGeneration();
    else sendMessage();
  };

  function generateId() {
    return 'r_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 9);
  }

  async function sendMessage() {
    const raw = userInput.value.trim();
    if (!raw || isGenerating) return;

    const ws = document.getElementById('welcomeScreen');
    if (ws) ws.remove();

    isGenerating = true;
    currentRequestId = generateId();
    sendStopBtn.classList.add('generating');

    appendUserMessage(raw);
    userInput.value = '';
    userInput.style.height = 'auto';

    const aiRow  = appendAiContainer();
    const bubble = aiRow.querySelector('.message-bubble');
    const cursor = createCursor();
    bubble.appendChild(cursor);
    scheduleScroll();

    const prompt = buildPrompt(raw);

    const params = {
      prompt,
      request_id:  currentRequestId,
      max_tokens:  parseInt(document.getElementById('maxTokens').value)     || 512,
      temperature: parseFloat(document.getElementById('temperature').value)  || 0.8,
      top_k:       parseInt(document.getElementById('topK').value)           || 50,
      top_p:       parseFloat(document.getElementById('topP').value)         || 0.9,
      rep_p:       parseFloat(document.getElementById('repPenalty').value)   || 1.1,
      seed:        getEffectiveSeed(),
      ignore_eos:  document.getElementById('ignoreEOS').checked,
      stream:      document.getElementById('streaming').checked,
    };

    abortController = new AbortController();

    try {
      const res = await fetch('/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(params),
        signal:  abortController.signal,
      });

      if (!res.ok) throw new Error('HTTP ' + res.status);

      if (params.stream) {
        await streamResponse(res, bubble, cursor, aiRow);
      } else {
        const data = await res.json();
        cursor.remove();
        renderIntoBubble(bubble, (data.generated_text || '').trim());
        if (data.stopped) appendStopped(aiRow);
      }
    } catch (err) {
      cursor.remove();
      if (err.name === 'AbortError') {
        if (!bubble.textContent.trim()) {
          bubble.innerHTML = '<span class="stopped-indicator">[stopped]</span>';
        } else {
          appendStopped(aiRow);
        }
      } else {
        bubble.innerHTML = '<span style="color:#e44">Error: ' + escHtml(err.message) + '</span>';
      }
    } finally {
      isGenerating = false;
      sendStopBtn.classList.remove('generating');
      abortController   = null;
      currentRequestId  = null;
      userInput.focus();
      scheduleScroll();
    }
  }

  /* ════════════════════════════════════════
     STREAMING — производительный вариант

     Вместо innerHTML на каждый токен мы:
     1) Накапливаем текст в строке `fullText`
     2) Перерисовываем DOM только через rAF
        (не чаще одного раза за кадр)
  ════════════════════════════════════════ */
  async function streamResponse(res, bubble, cursor, aiRow) {
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();

    let buf          = '';
    let fullText     = '';
    let renderPending = false;
    let cursorInDom  = true;
    let stopped      = false;

    function scheduleRender() {
      if (renderPending) return;
      renderPending = true;
      requestAnimationFrame(() => {
        renderPending = false;
        renderIntoBubble(bubble, fullText.trimStart());
        /* возвращаем курсор после перерисовки */
        if (cursorInDom) bubble.appendChild(cursor);
        scheduleScroll();
      });
    }

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const j = JSON.parse(line.slice(6));

            if (j.done) {
              if (j.finish_reason === 'stop') stopped = true;
              /* flush */
              buf = '';
              break;
            }

            if (j.text) {
              if (cursorInDom && fullText === '') {
                /* первый токен — убираем одиночный курсор */
              }
              fullText += j.text;
              scheduleRender();
            }
          } catch (_) { /* ignore malformed SSE */ }
        }
      }
    } finally {
      /* финальный рендер */
      cursorInDom = false;
      cursor.remove();
      renderIntoBubble(bubble, fullText.trim() || '(no response)');
      if (stopped) appendStopped(aiRow);
      scheduleScroll();
    }
  }

  function stopGeneration() {
    if (currentRequestId) {
      fetch('/stop', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ request_id: currentRequestId }),
      }).catch(() => {});
    }
    if (abortController) { abortController.abort(); abortController = null; }
  }

  /* ════════════════════════════════════════
     RENDER MARKDOWN → DOM
  ════════════════════════════════════════ */

  /**
   * Конвертируем текст в HTML и вставляем в bubble.
   * Вынесено в отдельную функцию чтобы использовать
   * и при стриминге, и при полном ответе.
   */
  function renderIntoBubble(bubble, text) {
    bubble.innerHTML = renderMarkdown(text);
  }

  /**
   * Полноценный рендер Markdown в HTML-строку.
   *
   * Порядок обработки важен:
   *   1) Fenced code blocks  (``` ... ```)
   *   2) Chapter dividers    (--- Глава X ---)
   *   3) Inline code         (`...`)
   *   4) Bold / Italic
   *   5) Перенос строк       (\n → <br>)  — но НЕ внутри pre
   */
  function renderMarkdown(text) {

    /* --- 1. Вырезаем fenced code blocks, заменяем на плейсхолдеры --- */
    const codeBlocks = [];
    let html = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const idx = codeBlocks.length;
      codeBlocks.push('<pre><code>' + escHtml(code.trimEnd()) + '</code></pre>');
      return '\x00CODE' + idx + '\x00';
    });

    /* --- 2. Chapter dividers: --- ТЕКСТ --- или --- глава 1 --- --- */
    /*  Паттерн: строка, которая начинается и заканчивается на ---    */
    /*  с любым текстом посередине (регистронезависимо)               */
    html = html.replace(/^[ \t]*---+[ \t]*(.+?)[ \t]*---+[ \t]*$/gm, (_, label) => {
      const safeLabel = escHtml(label.trim());
      return (
        '<div class="chapter-divider">' +
          '<div class="chapter-divider-line"></div>' +
          '<div class="chapter-divider-label">' + safeLabel + '</div>' +
          '<div class="chapter-divider-line"></div>' +
        '</div>'
      );
    });

    /* --- 3. Экранируем HTML вне code-плейсхолдеров --- */
    /*  Нам нужно экранировать только "живой" текст, не плейсхолдеры */
    /*  Разбиваем по плейсхолдерам и экранируем каждую часть         */
    const parts = html.split(/(\x00CODE\d+\x00)/);
    html = parts.map((part, i) => {
      /* нечётные части — плейсхолдеры, оставляем как есть */
      if (i % 2 === 1) return part;

      /* ---- inline code ---- */
      /* Сначала вырезаем inline code чтобы не трогать его содержимое */
      const inlineCodes = [];
      let p = part.replace(/`([^`\n]+)`/g, (_, code) => {
        const idx = inlineCodes.length;
        inlineCodes.push('<code>' + escHtml(code) + '</code>');
        return '\x01IC' + idx + '\x01';
      });

      /* ---- chapter divider блоки уже вставлены, экранировать их нельзя */
      /*  вырезаем их тоже                                                  */
      const divBlocks = [];
      p = p.replace(/(<div class="chapter-divider">[\s\S]*?<\/div>)/g, (_, blk) => {
        const idx = divBlocks.length;
        divBlocks.push(blk);
        return '\x02DIV' + idx + '\x02';
      });

      /* ---- bold / italic (простой однострочный вариант) ---- */
      p = p.replace(/\*\*([^*\n]+?)\*\*/g, (_, t) => '<strong>' + escHtml(t) + '</strong>');
      p = p.replace(/\*([^*\n]+?)\*/g,     (_, t) => '<em>'      + escHtml(t) + '</em>');

      /* ---- переносы строк ---- */
      /* Заменяем \n\n на параграфный отступ, одиночный \n — на <br> */
      p = p.replace(/\n{2,}/g, '<br><br>');
      p = p.replace(/\n/g, '<br>');

      /* ---- восстанавливаем divider-блоки ---- */
      p = p.replace(/\x02DIV(\d+)\x02/g, (_, idx) => divBlocks[+idx]);

      /* ---- восстанавливаем inline code ---- */
      p = p.replace(/\x01IC(\d+)\x01/g, (_, idx) => inlineCodes[+idx]);

      return p;
    }).join('');

    /* --- 4. Восстанавливаем code blocks --- */
    html = html.replace(/\x00CODE(\d+)\x00/g, (_, idx) => codeBlocks[+idx]);

    return html;
  }

  /* ════════════════════════════════════════
     MESSAGES
  ════════════════════════════════════════ */
  function appendUserMessage(text) {
    const row = el('div', 'message-row user');
    row.innerHTML =
      '<div class="message-label">You</div>' +
      '<div class="message-bubble">' + escHtml(text) + '</div>';
    chatContainer.appendChild(row);
    scheduleScroll();
  }

  function appendAiContainer() {
    const row = el('div', 'message-row ai');
    row.innerHTML =
      '<div class="message-label">Assistant</div>' +
      '<div class="message-bubble"></div>';
    chatContainer.appendChild(row);
    scheduleScroll();
    return row;
  }

  function appendStopped(row) {
    const d = el('div', 'stopped-indicator');
    d.textContent = '[generation stopped]';
    row.appendChild(d);
  }

  function createCursor() {
    return el('span', 'cursor-blink');
  }

  /* ════════════════════════════════════════
     CLEAR
  ════════════════════════════════════════ */
  window.clearChat = function () {
    chatContainer.innerHTML = '';
    const ws = el('div', 'welcome-screen');
    ws.id = 'welcomeScreen';
    ws.innerHTML =
      '<h1>XAI Engine</h1>' +
      '<p>High-performance local inference.<br>Open settings to configure parameters and prompt templates.</p>' +
      '<span class="hint">Enter — send &nbsp;·&nbsp; Shift+Enter — new line</span>';
    chatContainer.appendChild(ws);
  };

  /* ════════════════════════════════════════
     UTILS
  ════════════════════════════════════════ */
  function el(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  function escHtml(t) {
    return String(t)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function escAttr(t) {
    return String(t)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  /* ════════════════════════════════════════
     INIT
  ════════════════════════════════════════ */
  parseTemplateVariables();
  userInput.focus();

})();
</script>
</body>
</html>
)HTML";
}

} // namespace webui
} // namespace xai