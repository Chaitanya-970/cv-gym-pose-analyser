"""FastAPI application and standalone local server for Raw Squat Video Review Workspace.

Attribution & Patterns:
- Built with FastAPI (MIT License, Copyright (c) 2018 Sebastián Ramírez)
- UI and dark-themed styling designed for single-user localhost human video review.
- Strict XSS protection using DOM textContent exclusively for untrusted content.
- Video playback directly streaming raw original uploaded videos with no inference overlays.
- Pure human review: No YOLO, Torch, Pose Pipeline, or Repetition Counter imports.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional, Union

# Deferred FastAPI imports to allow standard-library CLI inspection
try:
    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
except ImportError:
    FastAPI = None  # type: ignore
    File = None  # type: ignore
    Form = None  # type: ignore
    HTTPException = None  # type: ignore
    Query = None  # type: ignore
    UploadFile = None  # type: ignore
    status = None  # type: ignore
    FileResponse = None  # type: ignore
    HTMLResponse = None  # type: ignore
    JSONResponse = None  # type: ignore

try:
    from gym_pose.gym_analyser.review_store import (
        ALLOWED_CYCLES,
        ALLOWED_EVENT_CLASSES,
        ALLOWED_EXTENSIONS,
        ALLOWED_GROUPS,
        ALLOWED_INTENTS,
        ALLOWED_SCOPES,
        DEFAULT_MAX_UPLOAD_BYTES,
        CaseCorruptedError,
        CaseNotFoundError,
        InvalidCaseIdError,
        ReviewStore,
        is_valid_uuid4,
    )
except ImportError:
    try:
        from .review_store import (  # type: ignore
            ALLOWED_CYCLES,
            ALLOWED_EXTENSIONS,
            ALLOWED_GROUPS,
            ALLOWED_INTENTS,
            ALLOWED_SCOPES,
            DEFAULT_MAX_UPLOAD_BYTES,
            CaseCorruptedError,
            CaseNotFoundError,
            InvalidCaseIdError,
            ReviewStore,
            is_valid_uuid4,
        )
    except ImportError:
        from gym_analyser.review_store import (  # type: ignore
            ALLOWED_CYCLES,
            ALLOWED_EXTENSIONS,
            ALLOWED_GROUPS,
            ALLOWED_INTENTS,
            ALLOWED_SCOPES,
            DEFAULT_MAX_UPLOAD_BYTES,
            CaseCorruptedError,
            CaseNotFoundError,
            InvalidCaseIdError,
            ReviewStore,
            is_valid_uuid4,
        )

try:
    from gym_pose.gym_analyser.review_store import DEFAULT_STORAGE_ROOT  # type: ignore
except (ImportError, AttributeError):
    try:
        from .review_store import DEFAULT_STORAGE_ROOT  # type: ignore
    except (ImportError, AttributeError):
        try:
            from gym_analyser.review_store import DEFAULT_STORAGE_ROOT  # type: ignore
        except (ImportError, AttributeError):
            DEFAULT_STORAGE_ROOT = (
                Path(__file__).resolve().parent.parent / "output" / "human_reviews"
            )

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Raw Squat Video Review Workspace</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      line-height: 1.5;
      padding: 24px 16px;
    }
    .container {
      max-width: 1240px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    header {
      text-align: center;
      margin-bottom: 4px;
    }
    header h1 {
      font-size: 1.85rem;
      font-weight: 700;
      color: var(--primary);
    }
    header p {
      color: var(--text-muted);
      font-size: 0.95rem;
      margin-top: 4px;
    }
    .notice-box {
      background: #091322;
      border-left: 4px solid var(--warning);
      padding: 12px 16px;
      border-radius: 4px;
      font-size: 0.9rem;
      color: #e2e8f0;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .notice-box strong {
      color: var(--warning);
    }
    .layout-grid {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 960px) {
      .layout-grid {
        grid-template-columns: 1fr;
      }
    }
    .left-col, .right-col {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 18px 20px;
    }
    .card h2 {
      font-size: 1.15rem;
      font-weight: 600;
      margin-bottom: 12px;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .card h3 {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }
    .form-group label {
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-muted);
    }
    .form-row {
      display: flex;
      gap: 12px;
    }
    .form-row .form-group {
      flex: 1;
    }
    input[type="text"],
    input[type="number"],
    select,
    textarea {
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
      font-family: inherit;
      width: 100%;
    }
    input[type="text"]:focus,
    input[type="number"]:focus,
    select:focus,
    textarea:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 1px var(--primary);
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: var(--primary);
      color: #0f172a;
      font-weight: 600;
      border: none;
      border-radius: 6px;
      padding: 8px 16px;
      font-size: 0.9rem;
      cursor: pointer;
      text-decoration: none;
      transition: background 0.15s ease;
    }
    .btn:hover {
      background: var(--primary-hover);
    }
    .btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-secondary {
      background: #334155;
      color: var(--text);
    }
    .btn-secondary:hover {
      background: #475569;
    }
    .btn-danger {
      background: var(--danger);
      color: #ffffff;
    }
    .btn-danger:hover {
      background: #dc2626;
    }
    .btn-sm {
      padding: 4px 10px;
      font-size: 0.8rem;
    }
    .btn-group {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .status-msg {
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 0.85rem;
      display: none;
      word-break: break-word;
      margin-top: 8px;
    }
    .status-success {
      background: rgba(34, 197, 94, 0.15);
      border: 1px solid var(--success);
      color: #86efac;
    }
    .status-error {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid var(--danger);
      color: #fca5a5;
    }
    .status-info {
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid var(--primary);
      color: #bae6fd;
    }
    /* Case List Table */
    .table-container {
      max-height: 380px;
      overflow-y: auto;
      border: 1px solid var(--card-border);
      border-radius: 6px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--card-border);
    }
    th {
      background: #091322;
      color: var(--text-muted);
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    tbody tr {
      cursor: pointer;
      transition: background 0.1s ease;
    }
    tbody tr:hover {
      background: #253349;
    }
    tbody tr.selected-case {
      background: #1e3a5f;
      outline: 1px solid var(--primary);
    }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .badge-unset {
      background: #334155;
      color: var(--text-muted);
    }
    .badge-set {
      background: rgba(34, 197, 94, 0.2);
      color: #86efac;
      border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .badge-scope {
      background: rgba(56, 189, 248, 0.15);
      color: #7dd3fc;
    }
    .badge-group {
      background: rgba(245, 158, 11, 0.15);
      color: #fcd34d;
    }
    /* Video section */
    .video-wrapper {
      background: #000000;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      aspect-ratio: 16 / 9;
      max-height: 440px;
      border: 1px solid var(--card-border);
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .video-actions {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 10px;
    }
    .video-actions a {
      color: var(--primary);
      font-size: 0.88rem;
      text-decoration: underline;
    }
    .codec-note {
      font-size: 0.8rem;
      color: var(--text-muted);
      background: #091322;
      padding: 8px 12px;
      border-radius: 4px;
      border: 1px solid #1e293b;
    }
    /* Metadata HUD */
    .hud-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      font-size: 0.82rem;
      background: #0f172a;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid var(--card-border);
    }
    .hud-item {
      display: flex;
      flex-direction: column;
      gap: 2px;
      overflow: hidden;
    }
    .hud-label {
      color: var(--text-muted);
      font-size: 0.75rem;
      text-transform: uppercase;
    }
    .hud-val {
      color: var(--text);
      font-weight: 500;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: monospace;
    }
    /* Checkbox rows */
    .checkbox-label {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.88rem;
      color: var(--text);
      cursor: pointer;
      user-select: none;
    }
    .checkbox-label input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--primary);
    }
    /* Events table */
    .events-table-wrapper {
      margin-top: 8px;
      margin-bottom: 12px;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      overflow-x: auto;
    }
    .events-table-wrapper table th {
      font-size: 0.78rem;
      padding: 6px 8px;
    }
    .events-table-wrapper table td {
      padding: 6px 8px;
    }
    .events-table-wrapper input[type="text"],
    .events-table-wrapper input[type="number"],
    .events-table-wrapper select {
      padding: 4px 6px;
      font-size: 0.82rem;
      min-width: 0;
    }
    .time-capture-cell {
      display: flex;
      gap: 4px;
      align-items: center;
    }
    .time-capture-cell input {
      flex: 1;
      min-width: 60px;
    }
    .btn-capture {
      padding: 3px 6px;
      font-size: 0.75rem;
      line-height: 1;
    }
    .btn-remove-event {
      padding: 2px 8px;
      font-size: 1rem;
      line-height: 1;
    }
    .btn:focus,
    button:focus {
      outline: 2px solid var(--primary);
      outline-offset: 1px;
    }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    .empty-state {
      text-align: center;
      padding: 24px 12px;
      color: var(--text-muted);
      font-size: 0.9rem;
    }
    .export-links {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
  </style>
</head>
<body>
  <div class="container">
    <a id="nav-back-analyser" href="/" class="nav-link" style="display:none; color: var(--primary); text-decoration: none; font-size: 0.9rem; margin-bottom: 8px;">&larr; Back to Pose Analyser</a>
    <header>
      <h1>Raw Squat Video Human Review Workspace</h1>
      <p>Independent ground truth annotation and review interface</p>
    </header>

    <div class="notice-box">
      <strong>Clear Review Protocol:</strong>
      <span>No model predictions are shown. Watch the raw video and enter observed counts and movement notes.</span>
    </div>

    <div class="layout-grid">
      <!-- Left Column: Upload, Cases List, Export -->
      <div class="left-col">
        <!-- Upload Section -->
        <div class="card">
          <h2>Upload Video</h2>
          <form id="upload-form" class="form-group">
            <div class="form-group">
              <label for="upload-file">Select Video (.mp4, .webm, .mov):</label>
              <input type="file" id="upload-file" accept=".mp4,.webm,.mov" required>
            </div>
            <div class="form-group">
              <label for="upload-group">Dataset Group:</label>
              <select id="upload-group">
                <option value="candidate" selected>candidate</option>
                <option value="development">development</option>
              </select>
            </div>
            <button type="submit" id="btn-upload" class="btn">Upload Video</button>
            <div id="upload-status" class="status-msg"></div>
          </form>
        </div>

        <!-- Cases List -->
        <div class="card">
          <h2>
            <span>Review Queue</span>
            <button type="button" id="btn-refresh-list" class="btn btn-secondary btn-sm">Refresh</button>
          </h2>
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Clip ID</th>
                  <th>Filename</th>
                  <th>Group</th>
                  <th>Count</th>
                  <th>Scope</th>
                </tr>
              </thead>
              <tbody id="cases-tbody">
                <tr><td colspan="5" class="empty-state">Loading cases...</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Export Section -->
        <div class="card">
          <h2>Export Ground Truth</h2>
          <div class="export-links">
            <a id="btn-export-counts" href="#" download="counts.json" class="btn btn-secondary btn-sm">
              Export Ground Truth (counts.json)
            </a>
            <a id="btn-export-manifest" href="#" download="manifest.json" class="btn btn-secondary btn-sm">
              Export Full Manifest (manifest.json)
            </a>
          </div>
        </div>
      </div>

      <!-- Right Column: Player, HUD, Form -->
      <div class="right-col">
        <!-- Video Player Card -->
        <div class="card">
          <h2>Video Player</h2>
          <div class="video-wrapper">
            <video id="player" controls playsinline preload="metadata">
              Your browser does not support HTML5 video.
            </video>
          </div>
          <div class="video-actions">
            <div>
              <a id="raw-download-link" href="#" target="_blank" download>
                Download Raw Video (video?download=1)
              </a>
            </div>
            <p class="codec-note">
              If your browser cannot decode this video codec directly (e.g. HEVC/H.265 or QuickTime),
              click the raw download link above to inspect the original video in VLC or your system player.
            </p>
          </div>
        </div>

        <!-- Case Metadata HUD -->
        <div class="card">
          <h3>Clip Metadata HUD</h3>
          <div class="hud-grid">
            <div class="hud-item">
              <span class="hud-label">Clip ID</span>
              <span id="hud-case-id" class="hud-val">-</span>
            </div>
            <div class="hud-item">
              <span class="hud-label">Original Filename</span>
              <span id="hud-filename" class="hud-val">-</span>
            </div>
            <div class="hud-item">
              <span class="hud-label">Dataset Group</span>
              <span id="hud-group" class="hud-val">-</span>
            </div>
            <div class="hud-item">
              <span class="hud-label">Source SHA256</span>
              <span id="hud-sha256" class="hud-val" title="">-</span>
            </div>
            <div class="hud-item">
              <span class="hud-label">Created At</span>
              <span id="hud-created" class="hud-val">-</span>
            </div>
            <div class="hud-item">
              <span class="hud-label">Updated At</span>
              <span id="hud-updated" class="hud-val">-</span>
            </div>
          </div>
        </div>

        <!-- Review Form Card -->
        <div class="card">
          <h2>Human Review Annotation</h2>
          <form id="review-form">
            <div class="form-row">
              <div class="form-group">
                <label for="review-count">Observed Human Rep Count:</label>
                <div class="btn-group">
                  <input type="text" id="review-count" inputmode="numeric" placeholder="None (unreviewed)">
                  <button type="button" id="btn-count-zero" class="btn btn-secondary btn-sm">Set to 0</button>
                  <button type="button" id="btn-count-clear" class="btn btn-secondary btn-sm">Clear (Unset)</button>
                </div>
              </div>
              <div class="form-group">
                <label for="review-scope">Scope Classification:</label>
                <select id="review-scope">
                  <option value="in_scope">in_scope</option>
                  <option value="out_of_scope">out_of_scope</option>
                  <option value="uncertain">uncertain</option>
                </select>
              </div>
            </div>

            <div class="form-group">
              <label for="review-reviewer">Reviewer Name / Identifier:</label>
              <input type="text" id="review-reviewer" maxlength="100" placeholder="e.g. Alex (reviewer)">
            </div>

            <div class="form-group">
              <label class="checkbox-label">
                <input type="checkbox" id="review-raw-ack">
                <span>I acknowledge that I reviewed the raw video directly.</span>
              </label>
            </div>

            <div class="form-group">
              <label for="review-preds-seen">Prior Predictions Exposure (Did you see model predictions before this review?):</label>
              <select id="review-preds-seen">
                <option value="" selected>-- Please select --</option>
                <option value="false">No (I have not seen predictions)</option>
                <option value="true">Yes (Predictions previously seen)</option>
              </select>
            </div>

            <div class="form-group">
              <label for="review-notes">Qualitative Movement Notes:</label>
              <textarea id="review-notes" rows="3" maxlength="5000" placeholder="Observations regarding depth, camera angle, setup, lighting, or form..."></textarea>
            </div>

            <div class="form-group">
              <label id="events-table-heading">Movement Temporal Events (Repetitions & Actions):</label>
              <div class="events-table-wrapper">
                <table aria-labelledby="events-table-heading">
                  <thead>
                    <tr>
                      <th style="min-width: 90px;">Event ID</th>
                      <th style="min-width: 140px;">Event Class</th>
                      <th style="min-width: 110px;">Start (s)</th>
                      <th style="min-width: 110px;">Turnaround (s)</th>
                      <th style="min-width: 110px;">End (s)</th>
                      <th style="min-width: 140px;">Posture Observations</th>
                      <th style="min-width: 140px;">Uncertainty Notes</th>
                      <th style="width: 40px;"><span class="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody id="events-tbody">
                  </tbody>
                </table>
              </div>
              <button type="button" id="btn-add-event" class="btn btn-secondary btn-sm" aria-label="Add new temporal event">+ Add Event</button>

              <template id="event-row-template">
                <tr>
                  <td>
                    <input type="text" class="event-id" placeholder="evt_1" aria-label="Event ID" title="Event ID">
                  </td>
                  <td>
                    <select class="event-class" aria-label="Event Class" title="Event Class">
                      <option value="">Choose an observation...</option>
                      <option value="completed_rep">completed_rep</option>
                      <option value="setup_or_weight_return">setup_or_weight_return</option>
                      <option value="partial_or_unfinished">partial_or_unfinished</option>
                      <option value="ambiguous_or_unclear">ambiguous_or_unclear</option>
                    </select>
                  </td>
                  <td>
                    <div class="time-capture-cell">
                      <input type="number" step="any" min="0" class="event-start" data-action="capture-time" data-target="start" placeholder="0.0" aria-label="Start time (s)" title="Start time in seconds">
                      <button type="button" class="btn btn-secondary btn-sm btn-capture" data-action="capture-time" data-target="start" title="Capture current video time for start" aria-label="Capture start time">&darr;</button>
                    </div>
                  </td>
                  <td>
                    <div class="time-capture-cell">
                      <input type="number" step="any" min="0" class="event-turnaround" data-action="capture-time" data-target="turnaround" placeholder="null" aria-label="Turnaround time (s)" title="Turnaround inflection time in seconds">
                      <button type="button" class="btn btn-secondary btn-sm btn-capture" data-action="capture-time" data-target="turnaround" title="Capture current video time for turnaround" aria-label="Capture turnaround time">&darr;</button>
                    </div>
                  </td>
                  <td>
                    <div class="time-capture-cell">
                      <input type="number" step="any" min="0" class="event-end" data-action="capture-time" data-target="end" placeholder="0.0" aria-label="End time (s)" title="End time in seconds">
                      <button type="button" class="btn btn-secondary btn-sm btn-capture" data-action="capture-time" data-target="end" title="Capture current video time for end" aria-label="Capture end time">&darr;</button>
                    </div>
                  </td>
                  <td>
                    <input type="text" class="event-posture event-obs" maxlength="5000" placeholder="Posture observations..." aria-label="Visible posture observations" title="Visible posture observations">
                  </td>
                  <td>
                    <input type="text" class="event-uncertainty" maxlength="5000" placeholder="Uncertainty notes..." aria-label="Uncertainty notes" title="Uncertainty notes">
                  </td>
                  <td>
                    <button type="button" class="btn btn-danger btn-sm btn-remove-event" title="Remove Event" aria-label="Remove Event">&times;</button>
                  </td>
                </tr>
              </template>
            </div>

            <div class="btn-group" style="margin-top: 16px;">
              <button type="submit" id="btn-save-review" class="btn">Save Review</button>
              <button type="button" id="btn-reload-case" class="btn btn-secondary">Reload Case</button>
            </div>

            <div id="review-status" class="status-msg"></div>
          </form>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Pure helper functions (exportable for testing)
    function resolveReviewUrl(path, currentPathname) {
      const p = currentPathname !== undefined ? currentPathname : (typeof window !== 'undefined' && window.location ? window.location.pathname : '');
      const cleanPath = path.startsWith('/') ? path : '/' + path;
      const isMounted = p === '/review' || p.startsWith('/review/');
      const prefix = isMounted ? '/review' : '';
      return prefix + cleanPath;
    }

    function parseStrictCount(str) {
      if (str === null || str === undefined) {
        return null;
      }
      const s = String(str).trim();
      if (s === '') {
        return null;
      }
      if (!/^\\d+$/.test(s)) {
        return NaN;
      }
      const n = Number(s);
      return Number.isSafeInteger(n) && n >= 0 ? n : NaN;
    }

    function sanitizeErrorMessage(msg) {
      if (!msg) return 'An error occurred.';
      let s = String(msg);
      // Redact Windows absolute paths (e.g. C:\\... or C:/...)
      s = s.replace(/[a-zA-Z]:[\\\\/][^:\\s"'\\n<>]+/g, '[redacted-path]');
      // Redact Unix absolute paths (e.g. /home/... or /tmp/...)
      s = s.replace(/(^|[\\s"'(])(\\/[\\w\\.-]+){2,}/g, '$1[redacted-path]');
      return s;
    }

    // State management
    let currentCaseId = null;
    let activeRequestGen = 0;
    let hasUnsavedChanges = false;
    let allCases = [];

    const hasDoc = typeof document !== 'undefined';

    // DOM references
    const casesTbody = hasDoc ? document.getElementById('cases-tbody') : null;
    const player = hasDoc ? document.getElementById('player') : null;
    const videoPlayer = player;
    if (typeof window !== 'undefined') {
      window.videoPlayer = player;
    }
    const rawDownloadLink = hasDoc ? document.getElementById('raw-download-link') : null;
    const uploadForm = hasDoc ? document.getElementById('upload-form') : null;
    const uploadFile = hasDoc ? document.getElementById('upload-file') : null;
    const uploadGroup = hasDoc ? document.getElementById('upload-group') : null;
    const btnUpload = hasDoc ? document.getElementById('btn-upload') : null;
    const uploadStatus = hasDoc ? document.getElementById('upload-status') : null;
    const btnRefreshList = hasDoc ? document.getElementById('btn-refresh-list') : null;
    const navBackAnalyser = hasDoc ? document.getElementById('nav-back-analyser') : null;
    const btnExportCounts = hasDoc ? (document.getElementById('btn-export-counts') || document.getElementById('export-counts-link')) : null;
    const btnExportManifest = hasDoc ? (document.getElementById('btn-export-manifest') || document.getElementById('export-manifest-link')) : null;

    // HUD references
    const hudCaseId = hasDoc ? document.getElementById('hud-case-id') : null;
    const hudFilename = hasDoc ? document.getElementById('hud-filename') : null;
    const hudGroup = hasDoc ? document.getElementById('hud-group') : null;
    const hudSha256 = hasDoc ? document.getElementById('hud-sha256') : null;
    const hudCreated = hasDoc ? document.getElementById('hud-created') : null;
    const hudUpdated = hasDoc ? document.getElementById('hud-updated') : null;

    // Form references
    const reviewForm = hasDoc ? document.getElementById('review-form') : null;
    const reviewCount = hasDoc ? document.getElementById('review-count') : null;
    const btnCountZero = hasDoc ? document.getElementById('btn-count-zero') : null;
    const btnCountClear = hasDoc ? document.getElementById('btn-count-clear') : null;
    const reviewScope = hasDoc ? document.getElementById('review-scope') : null;
    const reviewReviewer = hasDoc ? document.getElementById('review-reviewer') : null;
    const reviewRawAck = hasDoc ? document.getElementById('review-raw-ack') : null;
    const reviewPredsSeen = hasDoc ? document.getElementById('review-preds-seen') : null;
    const reviewNotes = hasDoc ? document.getElementById('review-notes') : null;
    const eventsTbody = hasDoc ? document.getElementById('events-tbody') : null;
    const btnAddEvent = hasDoc ? document.getElementById('btn-add-event') : null;
    const btnSaveReview = hasDoc ? document.getElementById('btn-save-review') : null;
    const btnReloadCase = hasDoc ? document.getElementById('btn-reload-case') : null;
    const reviewStatus = hasDoc ? document.getElementById('review-status') : null;

    // Status helpers using textContent strictly
    function showStatus(elem, text, kind) {
      if (!elem) return;
      elem.textContent = text;
      elem.className = 'status-msg status-' + kind;
      elem.style.display = 'block';
    }

    function clearStatus(elem) {
      if (!elem) return;
      elem.textContent = '';
      elem.style.display = 'none';
      elem.className = 'status-msg';
    }

    function showSafeError(elem, text) {
      showStatus(elem, sanitizeErrorMessage(text), 'error');
    }

    function setReviewFormDisabled(disabled) {
      if (!reviewForm) return;
      const elements = reviewForm.querySelectorAll('input, select, textarea, button');
      elements.forEach(el => {
        el.disabled = disabled;
      });
    }

    function markUnsaved() {
      if (currentCaseId) {
        hasUnsavedChanges = true;
      }
    }

    // Initial form disabled state (until a case is selected)
    if (hasDoc) {
      setReviewFormDisabled(true);
    }

    // Optional Return Navigation when mounted under /review/
    if (typeof window !== 'undefined' && window.location && (window.location.pathname === '/review' || window.location.pathname.startsWith('/review/'))) {
      const backLink = document.getElementById('nav-back-analyser');
      if (backLink) backLink.style.display = 'inline-block';
    }

    // Dynamic resolution for export links
    if (btnExportCounts) {
      btnExportCounts.href = resolveReviewUrl('/api/export/counts');
    }
    if (btnExportManifest) {
      btnExportManifest.href = resolveReviewUrl('/api/export/manifest');
    }

    // Fetch case list
    async function fetchCases() {
      try {
        const res = await fetch(resolveReviewUrl('/api/cases'));
        if (!res.ok) {
          throw new Error('Failed to load cases: ' + res.status);
        }
        const data = await res.json();
        allCases = data.cases || [];
        renderCasesTable();
      } catch (err) {
        if (!casesTbody) return;
        casesTbody.textContent = '';
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 5;
        td.className = 'empty-state';
        td.textContent = sanitizeErrorMessage('Error loading cases: ' + err.message);
        tr.appendChild(td);
        casesTbody.appendChild(tr);
      }
    }

    // Render cases table with strict textContent
    function renderCasesTable() {
      if (!casesTbody) return;
      casesTbody.textContent = '';
      if (allCases.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 5;
        td.className = 'empty-state';
        td.textContent = 'No cases found. Upload a video above to begin.';
        tr.appendChild(td);
        casesTbody.appendChild(tr);
        return;
      }

      allCases.forEach(c => {
        const tr = document.createElement('tr');
        const cid = c.case_id || c.id || c.clip_id || '';
        if (cid === currentCaseId) {
          tr.classList.add('selected-case');
        }
        tr.addEventListener('click', () => selectCase(cid));

        // Clip ID
        const tdId = document.createElement('td');
        tdId.textContent = cid.substring(0, 8) + '...';
        tdId.title = cid;

        // Filename
        const tdName = document.createElement('td');
        tdName.textContent = c.original_filename || '-';

        // Group
        const tdGroup = document.createElement('td');
        const groupSpan = document.createElement('span');
        groupSpan.className = 'badge badge-group';
        groupSpan.textContent = c.group || '-';
        tdGroup.appendChild(groupSpan);

        // Count
        const tdCount = document.createElement('td');
        const countSpan = document.createElement('span');
        const count = c.review ? c.review.human_count : c.human_count;
        if (count === null || count === undefined) {
          countSpan.className = 'badge badge-unset';
          countSpan.textContent = 'Unset';
        } else {
          countSpan.className = 'badge badge-set';
          countSpan.textContent = String(count);
        }
        tdCount.appendChild(countSpan);

        // Scope
        const tdScope = document.createElement('td');
        const scopeSpan = document.createElement('span');
        scopeSpan.className = 'badge badge-scope';
        scopeSpan.textContent = (c.review && c.review.scope) ? c.review.scope : (c.scope || 'in_scope');
        tdScope.appendChild(scopeSpan);

        tr.appendChild(tdId);
        tr.appendChild(tdName);
        tr.appendChild(tdGroup);
        tr.appendChild(tdCount);
        tr.appendChild(tdScope);
        casesTbody.appendChild(tr);
      });
    }

    // Select and load a case
    async function selectCase(targetCaseId) {
      if (!targetCaseId) return;

      if (currentCaseId && hasUnsavedChanges) {
        const discard = confirm('You have unsaved changes. Discard them?');
        if (!discard) {
          return;
        }
      }

      activeRequestGen++;
      const gen = activeRequestGen;
      currentCaseId = targetCaseId;
      hasUnsavedChanges = false;
      setReviewFormDisabled(true);
      renderCasesTable();
      clearStatus(reviewStatus);

      try {
        const res = await fetch(resolveReviewUrl('/api/cases/' + encodeURIComponent(targetCaseId)));
        if (gen !== activeRequestGen) return;

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || ('Failed to retrieve case details: ' + res.status));
        }
        const caseData = await res.json();
        if (gen !== activeRequestGen) return;

        const loadedId = caseData.case_id || caseData.id || caseData.clip_id;
        if (loadedId !== targetCaseId) {
          throw new Error('Case ID mismatch in server response.');
        }

        populateCaseView(caseData);
        hasUnsavedChanges = false;
        setReviewFormDisabled(false);
      } catch (err) {
        if (gen !== activeRequestGen) return;
        showSafeError(reviewStatus, 'Failed to load case: ' + err.message);
        setReviewFormDisabled(true);
      }
    }

    // Populate HUD, Video Player, and Form
    function populateCaseView(caseData) {
      const cid = caseData.case_id || caseData.id || caseData.clip_id || '';

      // Update HUD
      if (hudCaseId) hudCaseId.textContent = cid || '-';
      if (hudFilename) hudFilename.textContent = caseData.original_filename || '-';
      if (hudGroup) hudGroup.textContent = caseData.group || '-';
      const sha = caseData.source_sha256 || caseData.sha256 || '';
      if (hudSha256) {
        hudSha256.textContent = sha ? sha.substring(0, 16) + '...' : '-';
        hudSha256.title = sha;
      }
      if (hudCreated) hudCreated.textContent = caseData.created_at ? new Date(caseData.created_at).toLocaleString() : '-';
      if (hudUpdated) hudUpdated.textContent = caseData.updated_at ? new Date(caseData.updated_at).toLocaleString() : '-';

      // Update Video Player
      if (player) {
        const videoSrc = resolveReviewUrl('/api/cases/' + encodeURIComponent(cid) + '/video');
        player.src = videoSrc;
        player.load();
      }

      // Raw download link
      if (rawDownloadLink) {
        rawDownloadLink.href = resolveReviewUrl('/api/cases/' + encodeURIComponent(cid) + '/video?download=1');
      }

      // Populate Review Form
      const rev = caseData.review || caseData;
      if (reviewCount) {
        reviewCount.value = (rev.human_count !== null && rev.human_count !== undefined) ? String(rev.human_count) : '';
      }
      if (reviewScope) {
        reviewScope.value = rev.scope || 'in_scope';
      }
      if (reviewReviewer) {
        reviewReviewer.value = rev.reviewer || '';
      }
      if (reviewRawAck) {
        reviewRawAck.checked = Boolean(rev.raw_video_reviewed);
      }
      if (reviewPredsSeen) {
        if (rev.predictions_previously_seen === true) {
          reviewPredsSeen.value = 'true';
        } else if (rev.predictions_previously_seen === false) {
          reviewPredsSeen.value = 'false';
        } else {
          reviewPredsSeen.value = '';
        }
      }
      if (reviewNotes) {
        reviewNotes.value = rev.notes || '';
      }

      // Populate Events
      if (eventsTbody) {
        eventsTbody.textContent = '';
        const events = rev.events || [];
        events.forEach((ev, idx) => addEventRow(ev, idx));
      }

      hasUnsavedChanges = false;
    }

    // Dynamic Event Row Builder
    function addEventRow(ev = {}, idx = null) {
      if (!eventsTbody) return;
      const tr = document.createElement('tr');
      tr._originalEvent = JSON.parse(JSON.stringify(ev));
      tr._legacyEvent = Object.prototype.hasOwnProperty.call(ev, 'event_index') && !ev.event_id;

      // 1. Event ID
      const tdId = document.createElement('td');
      const inId = document.createElement('input');
      inId.type = 'text';
      inId.className = 'event-id';
      inId.setAttribute('aria-label', 'Event ID');
      inId.setAttribute('title', 'Event ID');
      let defaultId = '';
      if (ev.event_id) {
        defaultId = String(ev.event_id);
      } else if (idx !== null && idx !== undefined) {
        defaultId = 'evt_' + (idx + 1);
      } else {
        const rowCount = eventsTbody.querySelectorAll ? eventsTbody.querySelectorAll('tr').length : 0;
        const used = new Set(Array.from(eventsTbody.querySelectorAll('.event-id')).map(input => input.value));
        let number = rowCount + 1;
        while (used.has('evt_' + number)) number++;
        defaultId = 'evt_' + number;
      }
      inId.value = defaultId;
      inId.placeholder = defaultId;
      inId.addEventListener('input', markUnsaved);
      tdId.appendChild(inId);

      // 2. Event Class Select
      const tdClass = document.createElement('td');
      const selClass = document.createElement('select');
      selClass.className = 'event-class';
      selClass.setAttribute('aria-label', 'Event Class');
      selClass.setAttribute('title', 'Event Class');
      const classOptions = [
        'completed_rep',
        'setup_or_weight_return',
        'partial_or_unfinished',
        'ambiguous_or_unclear'
      ];
      let selectedClass = ev.event_class && classOptions.includes(ev.event_class) ? ev.event_class : '';
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = tr._legacyEvent
        ? 'Legacy observation (preserved unless explicitly reclassified)'
        : 'Choose an observation...';
      placeholder.selected = !selectedClass;
      selClass.appendChild(placeholder);
      classOptions.forEach(optVal => {
        const opt = document.createElement('option');
        opt.value = optVal;
        opt.textContent = optVal;
        if (optVal === selectedClass) opt.selected = true;
        selClass.appendChild(opt);
      });
      selClass.addEventListener('change', markUnsaved);
      tdClass.appendChild(selClass);

      // Helper for time capture cell
      function createTimeCell(className, targetName, initialVal, placeholder, label) {
        const td = document.createElement('td');
        const wrapper = document.createElement('div');
        wrapper.className = 'time-capture-cell';

        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.min = '0';
        input.className = className;
        input.setAttribute('data-action', 'capture-time');
        input.setAttribute('data-target', targetName);
        input.setAttribute('aria-label', label);
        input.setAttribute('title', label);
        input.placeholder = placeholder;
        if (initialVal !== undefined && initialVal !== null && initialVal !== '') {
          input.value = String(initialVal);
        } else {
          input.value = '';
        }
        input.addEventListener('input', markUnsaved);

        const btnCapture = document.createElement('button');
        btnCapture.type = 'button';
        btnCapture.className = 'btn btn-secondary btn-sm btn-capture';
        btnCapture.setAttribute('data-action', 'capture-time');
        btnCapture.setAttribute('data-target', targetName);
        btnCapture.setAttribute('title', 'Capture video time for ' + targetName);
        btnCapture.setAttribute('aria-label', 'Capture ' + targetName + ' time');
        btnCapture.textContent = '\u2193';
        btnCapture.addEventListener('click', () => {
          const vid = player || (typeof document !== 'undefined' ? document.getElementById('player') : null);
          if (vid && Number.isFinite(vid.currentTime)) {
            const captured = Number(vid.currentTime.toFixed(3));
            input.value = String(captured);
            markUnsaved();
          }
        });

        wrapper.appendChild(input);
        wrapper.appendChild(btnCapture);
        td.appendChild(wrapper);
        return { td, input };
      }

      // 3. Start Time
      const startCell = createTimeCell('event-start', 'start', ev.start_time_s, '0.0', 'Start time (s)');

      // 4. Turnaround Time
      const turnCell = createTimeCell('event-turnaround', 'turnaround', ev.turnaround_time_s, 'null', 'Turnaround time (s)');

      // 5. End Time
      const endCell = createTimeCell('event-end', 'end', ev.end_time_s, '0.0', 'End time (s)');

      // 6. Visible Posture Observations
      const tdPosture = document.createElement('td');
      const inPosture = document.createElement('input');
      inPosture.type = 'text';
      inPosture.maxLength = 5000;
      inPosture.className = 'event-posture event-obs';
      inPosture.setAttribute('aria-label', 'Visible posture observations');
      inPosture.setAttribute('title', 'Visible posture observations');
      inPosture.placeholder = 'Posture observations...';
      let obsText = '';
      if (typeof ev.visible_posture_observations === 'string') {
        obsText = ev.visible_posture_observations;
      } else if (ev.visible_posture_observations && typeof ev.visible_posture_observations === 'object') {
        obsText = JSON.stringify(ev.visible_posture_observations);
      } else if (ev.observation) {
        obsText = String(ev.observation);
      }
      inPosture.value = obsText;
      inPosture.addEventListener('input', markUnsaved);
      tdPosture.appendChild(inPosture);

      // 7. Uncertainty Notes
      const tdUncertainty = document.createElement('td');
      const inUncertainty = document.createElement('input');
      inUncertainty.type = 'text';
      inUncertainty.maxLength = 5000;
      inUncertainty.className = 'event-uncertainty';
      inUncertainty.setAttribute('aria-label', 'Uncertainty notes');
      inUncertainty.setAttribute('title', 'Uncertainty notes');
      inUncertainty.placeholder = 'Uncertainty notes...';
      inUncertainty.value = ev.uncertainty_notes ? String(ev.uncertainty_notes) : '';
      inUncertainty.addEventListener('input', markUnsaved);
      tdUncertainty.appendChild(inUncertainty);

      // 8. Remove Button
      const tdRemove = document.createElement('td');
      const btnRemove = document.createElement('button');
      btnRemove.type = 'button';
      btnRemove.className = 'btn btn-danger btn-sm btn-remove-event';
      btnRemove.textContent = '\u00D7';
      btnRemove.title = 'Remove Event';
      btnRemove.setAttribute('aria-label', 'Remove Event');
      btnRemove.addEventListener('click', () => {
        tr.remove();
        markUnsaved();
      });
      tdRemove.appendChild(btnRemove);

      tr.appendChild(tdId);
      tr.appendChild(tdClass);
      tr.appendChild(startCell.td);
      tr.appendChild(turnCell.td);
      tr.appendChild(endCell.td);
      tr.appendChild(tdPosture);
      tr.appendChild(tdUncertainty);
      tr.appendChild(tdRemove);

      tr._initialValues = JSON.stringify(Array.from(tr.querySelectorAll('input, select')).map(input => input.value));
      eventsTbody.appendChild(tr);
    }

    // Attach form input listeners for unsaved change tracking
    if (reviewCount) reviewCount.addEventListener('input', markUnsaved);
    if (btnCountZero) {
      btnCountZero.addEventListener('click', () => {
        reviewCount.value = '0';
        markUnsaved();
      });
    }
    if (btnCountClear) {
      btnCountClear.addEventListener('click', () => {
        reviewCount.value = '';
        markUnsaved();
      });
    }
    if (reviewScope) reviewScope.addEventListener('change', markUnsaved);
    if (reviewReviewer) reviewReviewer.addEventListener('input', markUnsaved);
    if (reviewRawAck) reviewRawAck.addEventListener('change', markUnsaved);
    if (reviewPredsSeen) reviewPredsSeen.addEventListener('change', markUnsaved);
    if (reviewNotes) reviewNotes.addEventListener('input', markUnsaved);
    if (btnAddEvent) {
      btnAddEvent.addEventListener('click', () => {
        addEventRow();
        markUnsaved();
      });
    }

    // Upload Handler
    if (uploadForm) {
      uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const files = uploadFile.files;
        if (!files || files.length === 0) {
          showSafeError(uploadStatus, 'Please select a video file first.');
          return;
        }

        const file = files[0];
        const group = uploadGroup.value;

        btnUpload.disabled = true;
        showStatus(uploadStatus, 'Uploading ' + file.name + '...', 'info');

        try {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('group', group);

          const res = await fetch(resolveReviewUrl('/api/upload'), {
            method: 'POST',
            body: formData
          });

          if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || ('Upload failed with status ' + res.status));
          }

          const newCase = await res.json();
          const newId = newCase.case_id || newCase.id || newCase.clip_id;
          showStatus(uploadStatus, 'Upload complete: ' + (newCase.original_filename || file.name), 'success');
          uploadFile.value = '';

          await fetchCases();
          if (newId) {
            selectCase(newId);
          }
        } catch (err) {
          showSafeError(uploadStatus, 'Upload error: ' + err.message);
        } finally {
          btnUpload.disabled = false;
        }
      });
    }

    // Save Review Handler
    if (reviewForm) {
      reviewForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentCaseId) {
          showSafeError(reviewStatus, 'Please select a clip from the queue before saving.');
          return;
        }
        const savingCaseId = currentCaseId;
        const savingGen = activeRequestGen;

        // Parse and validate human count with strict integer checking
        const countVal = parseStrictCount(reviewCount.value);
        if (Number.isNaN(countVal)) {
          showSafeError(reviewStatus, 'Observed count must be an integer >= 0, or cleared for unreviewed.');
          return;
        }
        const humanCount = countVal;

        const reviewer = reviewReviewer.value.trim();
        const rawReviewed = reviewRawAck.checked;

        let predsSeen = null;
        if (reviewPredsSeen.value === 'true') {
          predsSeen = true;
        } else if (reviewPredsSeen.value === 'false') {
          predsSeen = false;
        } else {
          predsSeen = null;
        }

        // Prior predictions exposure validation
        if (humanCount !== null && predsSeen === null) {
          showSafeError(reviewStatus, 'Please indicate whether you have previously seen model predictions for this clip.');
          return;
        }

        // Co-dependency validation when count is set
        if (humanCount !== null) {
          if (!reviewer) {
            showSafeError(reviewStatus, 'Reviewer name is required when human count is set.');
            return;
          }
          if (!rawReviewed) {
            showSafeError(reviewStatus, 'You must acknowledge having reviewed the raw video directly.');
            return;
          }
        }

        const scope = reviewScope.value;
        const notes = reviewNotes.value;

        // Events collection with finite timestamp validation
        const events = [];
        const eventRows = eventsTbody && eventsTbody.querySelectorAll ? eventsTbody.querySelectorAll('tr') : [];
        const seenEventIds = new Set();

        for (let i = 0; i < eventRows.length; i++) {
          const row = eventRows[i];
          const idInput = row.querySelector('.event-id');
          const classSelect = row.querySelector('.event-class');
          const startInput = row.querySelector('.event-start');
          const turnInput = row.querySelector('.event-turnaround');
          const endInput = row.querySelector('.event-end');
          const postureInput = row.querySelector('.event-posture') || row.querySelector('.event-obs');
          const uncInput = row.querySelector('.event-uncertainty');

          const unchangedLegacy = row._legacyEvent &&
            JSON.stringify(Array.from(row.querySelectorAll('input, select')).map(input => input.value)) === row._initialValues;
          if (unchangedLegacy) {
            events.push(JSON.parse(JSON.stringify(row._originalEvent)));
            continue;
          }
          const eventId = idInput && idInput.value ? idInput.value.trim() : '';
          if (!eventId || seenEventIds.has(eventId)) {
            showSafeError(reviewStatus, 'Event ' + (i + 1) + ': Supply a unique event ID.');
            return;
          }
          seenEventIds.add(eventId);
          const eventClass = classSelect ? classSelect.value : '';
          if (!eventClass) {
            showSafeError(reviewStatus, 'Event ' + (i + 1) + ': Choose an observation explicitly; unknown/ambiguous is allowed.');
            return;
          }

          // Timestamps parsing and validation (empty string -> null)
          let startVal = null;
          if (startInput && startInput.value.trim() !== '') {
            startVal = Number(startInput.value);
            if (!Number.isFinite(startVal) || startVal < 0) {
              showSafeError(reviewStatus, 'Event ' + (i + 1) + ': Start time must be a finite number >= 0.');
              return;
            }
          }

          let turnVal = null;
          if (turnInput && turnInput.value.trim() !== '') {
            turnVal = Number(turnInput.value);
            if (!Number.isFinite(turnVal) || turnVal < 0) {
              showSafeError(reviewStatus, 'Event ' + (i + 1) + ': Turnaround time must be a finite number >= 0.');
              return;
            }
          }

          let endVal = null;
          if (endInput && endInput.value.trim() !== '') {
            endVal = Number(endInput.value);
            if (!Number.isFinite(endVal) || endVal < 0) {
              showSafeError(reviewStatus, 'Event ' + (i + 1) + ': End time must be a finite number >= 0.');
              return;
            }
          }

          // Ordering checks
          if (startVal !== null && turnVal !== null && turnVal < startVal) {
            showSafeError(reviewStatus, 'Event ' + (i + 1) + ': Turnaround time cannot be earlier than start time.');
            return;
          }
          if (turnVal !== null && endVal !== null && endVal < turnVal) {
            showSafeError(reviewStatus, 'Event ' + (i + 1) + ': End time cannot be earlier than turnaround time.');
            return;
          }
          if (startVal !== null && endVal !== null && endVal < startVal) {
            showSafeError(reviewStatus, 'Event ' + (i + 1) + ': End time cannot be earlier than start time.');
            return;
          }

          let postureObs = postureInput && postureInput.value.trim() ? postureInput.value.trim() : null;
          const originalPosture = row._originalEvent && row._originalEvent.visible_posture_observations;
          if (originalPosture && typeof originalPosture === 'object' && postureInput.value === JSON.stringify(originalPosture)) {
            postureObs = JSON.parse(JSON.stringify(originalPosture));
          }
          const uncNotes = uncInput && uncInput.value.trim() ? uncInput.value.trim() : null;

          const evtObj = {
            event_id: eventId,
            event_class: eventClass,
            start_time_s: startVal,
            turnaround_time_s: turnVal,
            end_time_s: endVal,
            uncertainty_notes: uncNotes,
            visible_posture_observations: postureObs,

          };

          events.push(evtObj);
        }

        const payload = {
          human_count: humanCount,
          scope: scope,
          reviewer: reviewer,
          raw_video_reviewed: rawReviewed,
          predictions_previously_seen: predsSeen,
          notes: notes,
          events: events,
          events_schema_version: 1
        };

        setReviewFormDisabled(true);
        btnSaveReview.textContent = 'Saving...';
        clearStatus(reviewStatus);

        try {
          const res = await fetch(resolveReviewUrl('/api/cases/' + encodeURIComponent(savingCaseId) + '/review'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });

          if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || ('Save failed with status ' + res.status));
          }

          const updatedCase = await res.json();
          const updatedId = updatedCase.case_id || updatedCase.id || updatedCase.clip_id;

          // Asynchronous race guard: Ensure saved response matches currentCaseId
          if (savingGen !== activeRequestGen || savingCaseId !== currentCaseId || updatedId !== currentCaseId) {
            await fetchCases();
            return;
          }

          hasUnsavedChanges = false;
          showStatus(reviewStatus, 'Review saved successfully at ' + new Date().toLocaleTimeString(), 'success');
          populateCaseView(updatedCase);
          await fetchCases();
        } catch (err) {
          if (savingGen === activeRequestGen && savingCaseId === currentCaseId) {
            showSafeError(reviewStatus, 'Save error: ' + err.message);
          }
        } finally {
          if (savingGen === activeRequestGen && savingCaseId === currentCaseId) {
            setReviewFormDisabled(false);
            btnSaveReview.textContent = 'Save Review';
          }
        }
      });
    }

    // Reload Case Button
    if (btnReloadCase) {
      btnReloadCase.addEventListener('click', () => {
        if (currentCaseId) {
          if (hasUnsavedChanges) {
            const discard = confirm('You have unsaved changes. Discard them?');
            if (!discard) return;
          }
          hasUnsavedChanges = false;
          selectCase(currentCaseId);
        }
      });
    }

    // Refresh List Button
    if (btnRefreshList) {
      btnRefreshList.addEventListener('click', () => {
        fetchCases();
      });
    }

    // Initial load
    if (hasDoc) {
      fetchCases();
    }

    // Export helpers for testing environments (e.g. Node.js unit tests)
    if (typeof module !== 'undefined' && module.exports) {
      module.exports = {
        parseStrictCount,
        sanitizeErrorMessage,
        resolveReviewUrl,
      };
    }
  </script>
</body>
</html>
"""

MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


def _format_public_case(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure top-level convenience access for review fields while allowlisting public schema."""
    if not isinstance(case_data, dict):
        return {}
    res: Dict[str, Any] = {}

    for k in (
        "case_id",
        "original_filename",
        "extension",
        "source_sha256",
        "file_size_bytes",
        "group",
        "created_at",
        "updated_at",
        "history",
    ):
        if k in case_data:
            res[k] = case_data[k]

    # Preserve case_id if passed as id or clip_id
    if "case_id" not in res:
        for alt_k in ("id", "clip_id"):
            if alt_k in case_data:
                res["case_id"] = case_data[alt_k]
                break

    rev = case_data.get("review")
    if isinstance(rev, dict):
        clean_rev: Dict[str, Any] = {}
        for rk in (
            "human_count",
            "scope",
            "reviewer",
            "raw_video_reviewed",
            "predictions_previously_seen",
            "notes",
            "events",
        ):
            if rk in rev:
                clean_rev[rk] = rev[rk]
                res[rk] = rev[rk]
        res["review"] = clean_rev
    else:
        # Check if review fields were provided flat
        clean_rev = {}
        for rk in (
            "human_count",
            "scope",
            "reviewer",
            "raw_video_reviewed",
            "predictions_previously_seen",
            "notes",
            "events",
        ):
            if rk in case_data:
                clean_rev[rk] = case_data[rk]
                res[rk] = case_data[rk]
        if clean_rev:
            res["review"] = clean_rev

    if "source_sha256" in res and "sha256" not in res:
        res["sha256"] = res["source_sha256"]

    return res


def create_review_app(
    storage_root: Optional[Union[str, Path]] = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    **kwargs: Any,
) -> FastAPI:
    """Create and configure the FastAPI application for raw squat video human reviews.

    Args:
        storage_root: Optional directory path where cases are stored.
            Defaults to DEFAULT_STORAGE_ROOT from review_store.
        max_upload_bytes: Maximum allowed video upload size in bytes.
        **kwargs: May accept 'root' as an alias for storage_root.
            Any other kwargs raise TypeError.

    Returns:
        Configured FastAPI application instance.
    """
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Please install dependencies: pip install -r requirements-web.txt"
        )

    target_root = storage_root
    if "root" in kwargs:
        target_root = kwargs.pop("root")

    if kwargs:
        unexpected = list(kwargs.keys())[0]
        raise TypeError(f"Unexpected keyword argument: {unexpected}")

    if target_root is None:
        target_root = DEFAULT_STORAGE_ROOT

    # Validate max_upload_bytes is positive integer (reject bool, float, negative, zero)
    if (
        isinstance(max_upload_bytes, bool)
        or not isinstance(max_upload_bytes, int)
        or max_upload_bytes <= 0
    ):
        raise ValueError(
            f"max_upload_bytes must be a positive integer, got {max_upload_bytes!r}"
        )

    store = ReviewStore(root=target_root, max_upload_bytes=max_upload_bytes)

    app = FastAPI(
        title="Raw Squat Video Review Workspace",
        description="Local human review workspace for raw squat video annotation and ground truth collection.",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index_endpoint() -> HTMLResponse:
        """Serve the standalone dark-themed human review HTML/JS interface."""
        return HTMLResponse(content=INDEX_HTML)

    @app.get("/api/cases")
    async def list_cases_endpoint() -> Dict[str, Any]:
        """Return list of all existing review cases."""
        try:
            raw_cases = store.list_cases()
        except CaseCorruptedError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Integrity error in stored review cases.",
            ) from exc
        return {"cases": [_format_public_case(c) for c in raw_cases]}

    @app.post("/api/upload", status_code=status.HTTP_201_CREATED)
    async def upload_endpoint(
        file: UploadFile = File(...),
        group: str = Form("candidate"),
    ) -> Dict[str, Any]:
        """Upload raw squat video, stream to temporary file, and create review case."""
        temp_path: Optional[Path] = None
        try:
            original_name = file.filename or ""
            ext = Path(original_name).suffix.lower()

            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(ALLOWED_EXTENSIONS)}",
                )

            if group not in ALLOWED_GROUPS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid group '{group}'. Allowed groups: {sorted(ALLOWED_GROUPS)}",
                )

            temp_fd, temp_path_str = tempfile.mkstemp(suffix=ext)
            os.close(temp_fd)
            temp_path = Path(temp_path_str)

            total_bytes = 0
            chunk_size = 64 * 1024

            with open(temp_path, "wb") as out_f:
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_upload_bytes:
                        if temp_path.exists():
                            try:
                                temp_path.unlink()
                            except OSError:
                                pass
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds size limit of {max_upload_bytes} bytes.",
                        )
                    out_f.write(chunk)

            case_data = store.create_case(
                source_path=temp_path,
                original_name=original_name,
                group=group,
            )
            return _format_public_case(case_data)
        except HTTPException:
            raise
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err
        finally:
            try:
                await file.close()
            except Exception:
                pass
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @app.get("/api/cases/{case_id}")
    async def get_case_endpoint(case_id: str) -> Dict[str, Any]:
        """Return public case metadata for a given UUID4 case ID."""
        if not is_valid_uuid4(case_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid case ID format. Standard UUID4 required.",
            )
        try:
            return _format_public_case(store.get_case(case_id))
        except CaseCorruptedError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Case metadata or raw video integrity error.",
            ) from exc
        except (CaseNotFoundError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case '{case_id}' not found.",
            ) from exc
        except (InvalidCaseIdError, ValueError) as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err

    @app.post("/api/cases/{case_id}/review")
    async def save_review_endpoint(case_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and persist human review annotation for a case."""
        if not is_valid_uuid4(case_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid case ID format. Standard UUID4 required.",
            )
        # Normalize event_index if omitted in legacy event dictionaries
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            for idx, ev in enumerate(payload["events"]):
                if (
                    isinstance(ev, dict)
                    and "event_index" not in ev
                    and "event_id" not in ev
                    and "event_class" not in ev
                ):
                    ev["event_index"] = idx

        try:
            return _format_public_case(store.save_review(case_id, payload))
        except CaseCorruptedError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Case metadata or raw video integrity error.",
            ) from exc
        except (CaseNotFoundError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case '{case_id}' not found.",
            ) from exc
        except (InvalidCaseIdError, ValueError) as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err

    @app.get("/api/cases/{case_id}/video")
    async def get_case_video_endpoint(
        case_id: str,
        download: Optional[str] = Query(None),
    ) -> FileResponse:
        """Stream or download raw original video for a case."""
        if not is_valid_uuid4(case_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid case ID format. Standard UUID4 required.",
            )
        try:
            v_path = store.video_path(case_id)
        except CaseCorruptedError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Raw video integrity check failed.",
            ) from exc
        except (CaseNotFoundError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video for case '{case_id}' not found.",
            ) from exc
        except (InvalidCaseIdError, ValueError) as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err

        ext = v_path.suffix.lower()
        media_type = MEDIA_TYPES.get(ext, "application/octet-stream")
        filename = v_path.name

        is_download = download is not None and download.lower() not in ("0", "false", "")
        disposition = "attachment" if is_download else "inline"

        return FileResponse(
            path=str(v_path),
            media_type=media_type,
            filename=filename,
            headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
        )

    @app.get("/api/export/counts")
    async def export_counts_endpoint() -> JSONResponse:
        """Export minimal ground truth counts schema compatible with evaluation.py."""
        try:
            return JSONResponse(content=store.export_counts())
        except CaseCorruptedError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Integrity error in stored review cases.",
            ) from exc

    @app.get("/api/export/manifest")
    async def export_manifest_endpoint() -> JSONResponse:
        """Export comprehensive review manifest including history and movement events."""
        try:
            return JSONResponse(content=store.export_manifest())
        except CaseCorruptedError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Integrity error in stored review cases.",
            ) from exc

    return app


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for review workspace launcher."""
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser.review_web",
        description="Raw Squat Video Review Workspace: local single-user human review interface.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Loopback host (default: 127.0.0.1). LAN binding is restricted.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to listen on (default: 8001).",
    )
    parser.add_argument(
        "--storage-root",
        type=str,
        default=None,
        help="Directory for human reviews storage (default: gym_pose/output/human_reviews).",
    )
    parser.add_argument(
        "--max-upload-bytes",
        type=int,
        default=DEFAULT_MAX_UPLOAD_BYTES,
        help=f"Maximum video upload size in bytes (default: {DEFAULT_MAX_UPLOAD_BYTES}).",
    )
    return parser


def main() -> None:
    """CLI launcher entry point for the review web application."""
    parser = build_parser()
    args = parser.parse_args()

    # Loopback restriction check
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        sys.stderr.write(
            f"Security Error: Host '{args.host}' is not a local loopback interface. "
            "Only 127.0.0.1, localhost, or ::1 is permitted.\\n"
        )
        sys.exit(1)

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "Error: uvicorn is not installed. Please install optional dependencies: "
            "pip install -r requirements-web.txt\\n"
        )
        sys.exit(1)

    storage_root = Path(args.storage_root) if args.storage_root else None
    app = create_review_app(storage_root=storage_root, max_upload_bytes=args.max_upload_bytes)

    print(f"Starting Raw Squat Video Review Workspace on http://{args.host}:{args.port}")
    if storage_root:
        print(f"Storage root: {storage_root.resolve()}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
