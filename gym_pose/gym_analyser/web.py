"""FastAPI web application and local server for 2D Gym Pose Analyser.

Attribution & Upstream Patterns:
- Built with FastAPI (MIT License, Copyright (c) 2018 SebastiÃ¡n RamÃ­rez)
  https://github.com/fastapi/fastapi/blob/master/LICENSE
- Request Files & UploadFile chunked handling adapted from:
  https://fastapi.tiangolo.com/tutorial/request-files/
- Background tasks pattern adapted from:
  https://fastapi.tiangolo.com/tutorial/background-tasks/

This module provides:
- Single-user, localhost-only web interface for squat video evaluation
- Secure endpoints for video submission, progress tracking, results, and downloads
- Strict allowlist downloads preventing path-traversal or log/credential exposure
- Transparent 2D biomechanical feedback and recording guidance
- CLI launcher: python -m gym_analyser.web --host 127.0.0.1 --port 8000 --device cpu
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Deferred framework imports to permit clean module inspection
try:
    from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
except ImportError:
    # Optional dependency: will be available when requirements-web.txt is installed
    FastAPI = None
    BackgroundTasks = None
    File = None
    HTTPException = None
    UploadFile = None
    status = None
    FileResponse = None
    HTMLResponse = None
    JSONResponse = None

from .web_jobs import (
    ALLOWED_ARTIFACTS,
    DEFAULT_MAX_UPLOAD_BYTES,
    JobBusyError,
    JobManager,
    is_valid_job_id,
)

INDEX_HTML = """<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>2D Gym Pose Analyser</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --primary-focus: rgba(56, 189, 248, 0.4);
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --touch-min: 44px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    *:focus-visible {
      outline: 3px solid var(--primary);
      outline-offset: 2px;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      line-height: 1.5;
      padding: clamp(12px, 2.5vw, 24px) clamp(8px, 2vw, 16px);
      min-height: 100vh;
    }
    .container {
      max-width: 980px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    header { text-align: center; margin-bottom: 8px; }
    header h1 {
      font-size: clamp(1.4rem, 4vw, 1.85rem);
      font-weight: 700;
      color: var(--primary);
      line-height: 1.25;
    }
    header p {
      color: var(--text-muted);
      font-size: clamp(0.85rem, 2vw, 0.95rem);
      margin-top: 6px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: clamp(14px, 3vw, 20px);
      overflow: hidden;
    }
    .card h2 {
      font-size: clamp(1.05rem, 2.5vw, 1.15rem);
      font-weight: 600;
      margin-bottom: 12px;
      color: var(--text);
    }
    .notice-box {
      background: #0f172a;
      border-left: 4px solid var(--primary);
      padding: 12px 16px;
      border-radius: 4px;
      font-size: 0.88rem;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 8px;
      line-height: 1.45;
    }
    .notice-box strong { color: var(--text); }
    .upload-form { display: flex; flex-direction: column; gap: 14px; }
    .file-input-wrapper {
      border: 2px dashed var(--card-border);
      border-radius: 8px;
      padding: clamp(16px, 4vw, 28px);
      text-align: center;
      cursor: pointer;
      min-height: 110px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: border-color 0.2s, background-color 0.2s, transform 0.15s;
    }
    .file-input-wrapper:hover {
      border-color: var(--primary);
      background-color: rgba(56, 189, 248, 0.04);
    }
    .file-input-wrapper.drag-over {
      border-color: var(--primary);
      background-color: rgba(56, 189, 248, 0.12);
      transform: scale(1.01);
    }
    .file-input-wrapper:focus-visible {
      border-color: var(--primary);
      outline: 3px solid var(--primary);
      outline-offset: 2px;
      background-color: rgba(56, 189, 248, 0.08);
    }
    .sr-only-input {
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
    .file-label {
      cursor: pointer;
      color: var(--primary);
      font-weight: 600;
      font-size: 1rem;
      min-height: var(--touch-min);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 4px 12px;
    }
    .file-label:hover { text-decoration: underline; }
    .selected-filename {
      font-size: 0.88rem;
      color: var(--text-muted);
      word-break: break-all;
    }
    .btn {
      background: var(--primary);
      color: #0f172a;
      border: none;
      min-height: var(--touch-min);
      min-width: var(--touch-min);
      padding: 10px 22px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      transition: background 0.2s, opacity 0.2s, transform 0.1s;
      align-self: flex-start;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-align: center;
    }
    .btn:hover:not(:disabled) { background: var(--primary-hover); }
    .btn:active:not(:disabled) { transform: translateY(1px); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-secondary {
      background: #334155;
      color: var(--text);
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 500;
      min-height: var(--touch-min);
      padding: 10px 16px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s, color 0.2s;
    }
    .btn-secondary:hover { background: #475569; }
    .btn-seek {
      background: #1e293b;
      border: 1px solid var(--primary);
      color: var(--primary);
      min-height: var(--touch-min);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s, color 0.2s;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      line-height: 1.2;
    }
    .btn-seek:hover {
      background: var(--primary);
      color: #0f172a;
    }
    .review-nav-link {
      color: var(--primary);
      font-size: 0.9rem;
      text-decoration: none;
      border: 1px solid var(--card-border);
      padding: 8px 16px;
      border-radius: 6px;
      background: var(--card-bg);
      min-height: var(--touch-min);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s, border-color 0.2s;
    }
    .review-nav-link:hover {
      background: #334155;
      border-color: var(--primary);
    }
    .status-badge {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 4px;
      font-size: 0.82rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .status-queued { background: #0284c7; color: #fff; }
    .status-running { background: #d97706; color: #fff; animation: pulse 1.8s infinite; }
    .status-completed { background: #16a34a; color: #fff; }
    .status-incomplete { background: #eab308; color: #000; }
    .status-failed, .status-interrupted { background: #dc2626; color: #fff; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
    .hidden { display: none !important; }
    .status-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric-card {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 14px 16px;
      text-align: center;
    }
    .metric-value { font-size: 1.75rem; font-weight: 700; color: var(--primary); }
    .metric-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; }
    .table-responsive {
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      margin-top: 12px;
      border: 1px solid var(--card-border);
      border-radius: 6px;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      min-width: 600px;
    }
    .data-table th, .data-table td {
      border: 1px solid var(--card-border);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }
    .data-table th { background: #0f172a; color: var(--text-muted); font-weight: 600; }
    .data-table tr:nth-child(even) { background: #172033; }
    .media-container {
      display: flex;
      flex-direction: column;
      gap: 16px;
      margin-top: 16px;
    }
    .media-box {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .media-box img, .media-box video {
      width: 100%;
      max-height: 480px;
      object-fit: contain;
      border-radius: 4px;
      background: #000;
    }
    .media-caption { font-size: 0.82rem; color: var(--text-muted); }
    .downloads-wrapper { width: 100%; }
    .downloads-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }
    .alert-box {
      padding: 12px 16px;
      border-radius: 6px;
      font-size: 0.88rem;
      margin-top: 12px;
      line-height: 1.45;
    }
    .alert-danger { background: #450a0a; border: 1px solid #7f1d1d; color: #fca5a5; }
    .alert-warning { background: #451a03; border: 1px solid #78350f; color: #fde68a; }
    @media (max-width: 640px) {
      .btn { width: 100%; align-self: stretch; }
      .btn-secondary { width: 100%; text-align: center; }
      .status-header { flex-direction: column; align-items: flex-start; }
      .metrics-grid { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <nav class="review-nav" style="display: flex; justify-content: flex-end; margin-bottom: 8px;"><a href="/review/" class="review-nav-link" id="review-link">&rarr; Human Review Workspace (Label raw video BEFORE viewing model output)</a></nav>
    <header>
      <h1>2D Gym Pose Analyser</h1>
      <p>Single-user local squat repetition counter and 2D kinematic feedback</p>
    </header>

    <!-- Recording Guidance & Technical Notices -->
    <div class="card">
      <h2>Recording Guidance & Operational Constraints</h2>
      <div class="notice-box">
        <div><strong>Human Review:</strong> To reduce prediction-induced bias, review and record raw video repetitions in the <a href="/review/" style="color: var(--primary); text-decoration: underline;">Human Review Workspace</a> before running automated model analysis.</div>
        <div><strong>Camera Setup:</strong> Single-person, stationary camera at waist height perpendicular to movement plane, showing full body from head to feet in side view.</div>
        <div><strong>Kinematic Feedback:</strong> Measurements represent projected 2D coordinates from this single camera view. They do not constitute clinical safety evaluations or medical form judgements.</div>
        <div><strong>Uncounted Movements:</strong> Movements not meeting threshold criteria (such as stance setup, weight pickup, or weight return) are reported as uncounted attempts. An uncounted event does not indicate bad form.</div>
      </div>
    </div>

    <!-- Upload Form -->
    <div class="card" id="upload-card">
      <h2>Upload Squat Video</h2>
      <form class="upload-form" id="upload-form">
        <div class="file-input-wrapper" id="drop-zone" tabindex="0" role="button" aria-controls="video-input" aria-label="Upload squat video file (click, press Enter or Space, or drag and drop here)">
          <input type="file" id="video-input" name="file" accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v" class="sr-only-input" tabindex="-1">
          <label for="video-input" class="file-label" id="file-label">Click to select video file (or drag & drop here)</label>
          <div class="selected-filename" id="file-name-display" aria-live="polite">No file selected (Max 200MB: MP4, MOV, AVI, MKV, WebM, M4V)</div>
        </div>
        <button type="submit" class="btn" id="submit-btn" disabled>Analyze Video</button>
      </form>
      <div id="upload-error" class="alert-box alert-danger hidden" role="alert" aria-live="assertive"></div>
    </div>

    <!-- Active Job Status Panel -->
    <div class="card hidden" id="status-card" role="region" aria-label="Analysis Job Status">
      <div class="status-header">
        <h2>Analysis Job Status</h2>
        <span class="status-badge" id="status-badge" role="status" aria-live="polite"><span id="job-status">QUEUED</span></span>
      </div>
      <div><strong>Job ID:</strong> <span id="job-id-display"></span></div>
      <div><strong>File:</strong> <span id="job-filename-display"></span></div>
      <div id="job-error-box" class="alert-box alert-danger hidden" role="alert" aria-live="assertive"></div>
      <div id="job-warning-box" class="alert-box alert-warning hidden" role="alert" aria-live="polite"></div>
    </div>

    <!-- Analysis Results Section -->
    <div class="card hidden" id="results-card" role="region" aria-label="Analysis Results">
      <h2 id="results-title">Analysis Results</h2>
      <div id="results-announcement" class="sr-only" role="status" aria-live="polite"></div>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value" id="metric-reps">0</div>
          <div class="metric-label" id="metric-reps-label">Completed Repetitions</div>
        </div>
        <div class="metric-card">
          <div class="metric-value" id="metric-attempts">0</div>
          <div class="metric-label">Observed Movements</div>
        </div>
        <div class="metric-card">
          <div class="metric-value" id="metric-uncounted">0</div>
          <div class="metric-label">Uncounted Attempts</div>
        </div>
        <div class="metric-card">
          <div class="metric-value" id="metric-frames">0</div>
          <div class="metric-label">Processed Frames</div>
        </div>
      </div>

      <!-- Incomplete Warning if Applicable -->
      <div id="incomplete-note-box" class="alert-box alert-warning hidden" role="alert" aria-live="polite">
        <strong>Incomplete Processing Warning:</strong> <span id="incomplete-note-text"></span>
      </div>

      <div style="margin-top: 16px;">
        <h3>Movement observations</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted);">Comparisons describe measured counted cycles in this video. They do not grade technique or establish safety.</p>
        <div id="session-observations"></div>
      </div>

      <!-- Coaching Feedback Card -->
      <div id="coaching-feedback-card" style="margin-top: 18px; border: 1px solid var(--card-border); border-radius: 6px; padding: 14px; background: #0f172a;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
          <h3 style="margin: 0;">Actionable Coaching Feedback &amp; Telemetry</h3>
          <span id="coaching-status-badge" class="status-badge" style="background: #334155; color: #f8fafc; font-size: 0.75rem;">UNVALIDATED</span>
        </div>

        <div class="notice-box" style="margin-bottom: 12px; font-size: 0.82rem;">
          <div><strong>Clinical Disclaimer:</strong> <span id="coaching-clinical-disclaimer">These 2D kinematic measurements are descriptive planar observations, not clinical diagnostic assessments or technique grades.</span></div>
          <div><strong>Validation Status:</strong> <span id="coaching-validation-status">UNVALIDATED by human expert review on diverse unseen subjects.</span></div>
        </div>

        <div id="coaching-feedback-content">
          <div id="coaching-quality-box" style="margin-bottom: 12px; font-size: 0.85rem;">
            <strong>Recording Quality Assessment:</strong>
            <div id="coaching-quality-details" style="color: var(--text-muted); margin-top: 4px;"></div>
          </div>

          <div id="coaching-cues-box" style="margin-bottom: 12px;">
            <strong style="font-size: 0.88rem;">Per-Repetition Pacing &amp; Depth Cues:</strong>
            <div id="coaching-cues-list" style="margin-top: 6px; display: flex; flex-direction: column; gap: 8px;"></div>
          </div>

          <div id="coaching-uncounted-box" style="margin-bottom: 6px;">
            <strong style="font-size: 0.88rem;">Uncounted Movement Context &amp; Hypotheses:</strong>
            <div id="coaching-uncounted-list" style="margin-top: 6px; display: flex; flex-direction: column; gap: 6px;"></div>
          </div>
        </div>
      </div>

      <!-- Attempt Observations Table -->
      <div style="margin-top: 16px;">
        <h3>Observed Movement Events</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">
          Includes counted repetitions and uncounted events (e.g. setup, depth threshold not reached, tracking interruption).
        </p>
        <div class="table-responsive" tabindex="0" role="region" aria-label="Observed Movement Events table">
          <table class="data-table" id="attempts-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Status</th>
                <th>Reason</th>
                <th>Frames (Time)</th>
                <th>Min Knee Angle</th>
                <th>Explanation & Posture Telemetry</th>
              </tr>
            </thead>
            <tbody id="attempts-tbody"></tbody>
          </table>
        </div>
      </div>

      <!-- Visual Media: Angle Plot and Annotated Video -->
      <div class="media-container">
        <div class="media-box" id="plot-box">
          <h3>Knee Flexion Angle Time Series</h3>
          <img id="angle-plot-img" src="" alt="Knee Angle Time Series Plot">
          <div class="media-caption">Knee angle trajectory with configured up (extension) and down (inflection) thresholds.</div>
        </div>

        <div class="media-box" id="video-box">
          <h3>Annotated Video Output</h3>
          <div class="video-player-container" id="preview-player">
            <video id="annotated-video" controls preload="metadata" aria-label="Squat analysis preview video">
              <source id="video-source" src="" type="video/mp4">
              Your browser does not support HTML5 video playback.
            </video>
          </div>
          <div id="video-fallback-notice" class="notice-box" style="margin-top: 8px;">
            <strong>Browser Playback Notice:</strong> Video preview is rendered in OpenCV mp4v format. Some web browsers require H.264 streams and may display a blank player. If playback does not start, use the download links below to view the video in VLC or your local media player.
          </div>
        </div>
      </div>

      <!-- Downloadable Artifacts -->
      <div style="margin-top: 20px;">
        <h3>Exported Artifacts</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">Download raw telemetry and verified analysis outputs:</p>
        <div class="downloads-wrapper" id="download-links">
          <div class="downloads-grid" id="downloads-container"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;
    const ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'];
    const ALLOWED_ARTIFACT_NAMES = [
      'summary.json',
      'attempts.json',
      'posture_summary.json',
      'feedback_observations.json',
      'feedback_phases.json',
      'coaching_feedback.json',
      'angles.csv',
      'angle_plot.png',
      'annotated.mp4',
      'preview.mp4'
    ];

    const fileInput = document.getElementById('video-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const dropZone = document.getElementById('drop-zone');
    const submitBtn = document.getElementById('submit-btn');
    const uploadForm = document.getElementById('upload-form');
    const uploadError = document.getElementById('upload-error');

    const statusCard = document.getElementById('status-card');
    const statusBadge = document.getElementById('status-badge');
    const jobIdDisplay = document.getElementById('job-id-display');
    const jobFilenameDisplay = document.getElementById('job-filename-display');
    const jobErrorBox = document.getElementById('job-error-box');
    const jobWarningBox = document.getElementById('job-warning-box');

    const resultsCard = document.getElementById('results-card');
    const metricReps = document.getElementById('metric-reps');
    const metricAttempts = document.getElementById('metric-attempts');
    const metricUncounted = document.getElementById('metric-uncounted');
    const metricFrames = document.getElementById('metric-frames');
    const incompleteNoteBox = document.getElementById('incomplete-note-box');
    const incompleteNoteText = document.getElementById('incomplete-note-text');
    const attemptsTbody = document.getElementById('attempts-tbody');
    const anglePlotImg = document.getElementById('angle-plot-img');
    const annotatedVideo = document.getElementById('annotated-video');
    const videoSource = document.getElementById('video-source');
    const downloadsContainer = document.getElementById('downloads-container');

    let currentJobId = null;
    let pollInterval = null;
    let statusRequestVersion = 0;
    let resultsRequestVersion = 0;
    let pendingSeekCleanup = null;
    let selectedFile = null;
    let isSubmitting = false;

    function formatFileSize(bytes) {
      if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return '0 B';
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }

    function getFileExtension(filename) {
      if (!filename || typeof filename !== 'string') return '';
      const dotIdx = filename.lastIndexOf('.');
      if (dotIdx === -1) return '';
      return filename.slice(dotIdx).toLowerCase();
    }

    function validateClientFile(file) {
      if (!file) {
        return { valid: false, error: 'No file selected.' };
      }
      if (file.size === 0) {
        return { valid: false, error: 'Uploaded file is empty (0 bytes).' };
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        return {
          valid: false,
          error: 'File size (' + formatFileSize(file.size) + ') exceeds maximum upload limit of 200 MB.'
        };
      }
      const ext = getFileExtension(file.name);
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        return {
          valid: false,
          error: 'Unsupported video format "' + (ext || 'none') + '". Allowed formats: ' + ALLOWED_EXTENSIONS.join(', ')
        };
      }
      return { valid: true, error: null };
    }

    function handleFileSelection(file) {
      if (!file) {
        selectedFile = null;
        fileNameDisplay.textContent = 'No file selected (Max 200MB: MP4, MOV, AVI, MKV, WebM, M4V)';
        submitBtn.disabled = true;
        uploadError.classList.add('hidden');
        uploadError.textContent = '';
        return;
      }

      const validation = validateClientFile(file);
      if (!validation.valid) {
        selectedFile = null;
        uploadError.textContent = validation.error;
        uploadError.classList.remove('hidden');
        fileNameDisplay.textContent = 'Invalid file: ' + file.name + ' (' + formatFileSize(file.size) + ')';
        submitBtn.disabled = true;
        return;
      }

      selectedFile = file;
      uploadError.classList.add('hidden');
      uploadError.textContent = '';
      fileNameDisplay.textContent = file.name + ' (' + formatFileSize(file.size) + ')';
      submitBtn.disabled = false;
    }

    function clearPendingSeek() {
      if (pendingSeekCleanup) {
        try {
          pendingSeekCleanup();
        } catch (err) {}
        pendingSeekCleanup = null;
      }
    }

    function isValidTimestamp(val) {
      return typeof val === 'number' && Number.isFinite(val) && val >= 0;
    }

    function seekVideoToTime(targetTime) {
      if (!isValidTimestamp(targetTime)) return;
      const t = Number(targetTime);
      clearPendingSeek();

      if (!annotatedVideo) return;

      const applySeek = () => {
        let seekTo = t;
        if (typeof annotatedVideo.duration === 'number' && Number.isFinite(annotatedVideo.duration) && annotatedVideo.duration > 0) {
          seekTo = Math.max(0, Math.min(t, annotatedVideo.duration));
        } else {
          seekTo = Math.max(0, t);
        }
        try {
          annotatedVideo.currentTime = seekTo;
        } catch (e) {}
      };

      if (annotatedVideo.readyState >= 1 && typeof annotatedVideo.duration === 'number' && Number.isFinite(annotatedVideo.duration)) {
        applySeek();
      } else {
        let timerId = null;
        const onLoaded = () => {
          cleanup();
          applySeek();
        };
        const cleanup = () => {
          annotatedVideo.removeEventListener('loadedmetadata', onLoaded);
          if (timerId !== null) {
            clearTimeout(timerId);
            timerId = null;
          }
          if (pendingSeekCleanup === cleanup) {
            pendingSeekCleanup = null;
          }
        };

        annotatedVideo.addEventListener('loadedmetadata', onLoaded);
        timerId = setTimeout(() => {
          cleanup();
        }, 5000);
        pendingSeekCleanup = cleanup;
      }
    }

    window.addEventListener('beforeunload', clearPendingSeek);

    // Keyboard and click activation for dropZone
    dropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });

    dropZone.addEventListener('click', (e) => {
      if (e.target === fileInput || e.target.closest('label[for="video-input"]')) {
        return; // Let native label-for association activate the input without duplicate click
      }
      fileInput.click();
    });

    // Drag and drop cues and file handling
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'dragend'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
      const files = e.dataTransfer ? e.dataTransfer.files : null;
      if (files && files.length > 0) {
        try {
          fileInput.files = files;
        } catch (_) {}
        handleFileSelection(files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      } else if (!selectedFile) {
        handleFileSelection(null);
      }
    });

    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (isSubmitting) return;

      const fileToUpload = selectedFile || (fileInput.files && fileInput.files[0]);
      if (!fileToUpload) {
        uploadError.textContent = 'Please select a video file first.';
        uploadError.classList.remove('hidden');
        return;
      }

      const validation = validateClientFile(fileToUpload);
      if (!validation.valid) {
        uploadError.textContent = validation.error;
        uploadError.classList.remove('hidden');
        submitBtn.disabled = true;
        return;
      }

      isSubmitting = true;
      submitBtn.disabled = true;
      submitBtn.textContent = 'Uploading & Enqueueing...';
      uploadError.classList.add('hidden');
      uploadError.textContent = '';

      const formData = new FormData();
      formData.append('file', fileToUpload);

      try {
        const res = await fetch('/api/jobs', {
          method: 'POST',
          body: formData
        });

        if (res.status === 429) {
          const data = await res.json().catch(() => null);
          const msg = (data && data.detail) || 'Server is busy processing another analysis job. Please wait and try again.';
          uploadError.textContent = msg;
          uploadError.classList.remove('hidden');
          isSubmitting = false;
          submitBtn.disabled = false;
          submitBtn.textContent = 'Analyze Video';
          return;
        }

        if (res.status === 503) {
          const data = await res.json().catch(() => null);
          const msg = (data && data.detail) || 'Service is temporarily unavailable (503). Please try again in a few moments.';
          uploadError.textContent = msg;
          uploadError.classList.remove('hidden');
          isSubmitting = false;
          submitBtn.disabled = false;
          submitBtn.textContent = 'Analyze Video';
          return;
        }

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: 'Upload failed' }));
          const msg = errData.detail || 'Upload failed with status ' + res.status;
          uploadError.textContent = msg;
          uploadError.classList.remove('hidden');
          isSubmitting = false;
          submitBtn.disabled = false;
          submitBtn.textContent = 'Analyze Video';
          return;
        }

        const data = await res.json();
        currentJobId = data.job_id;
        history.replaceState(null, "", "?job=" + encodeURIComponent(currentJobId));
        startJobTracking(currentJobId, data.filename || fileToUpload.name);
        isSubmitting = false;
      } catch (err) {
        uploadError.textContent = err.message || 'Network error during video upload. Please check your connection and retry.';
        uploadError.classList.remove('hidden');
        isSubmitting = false;
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze Video';
      }
    });

    function startJobTracking(jobId, filename) {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
      clearPendingSeek();
      currentJobId = jobId;
      statusCard.classList.remove('hidden');
      resultsCard.classList.add('hidden');
      jobIdDisplay.textContent = jobId;
      jobFilenameDisplay.textContent = filename || 'Video';
      jobErrorBox.classList.add('hidden');
      jobErrorBox.textContent = '';
      jobWarningBox.classList.add('hidden');
      jobWarningBox.textContent = '';

      updateStatusBadge('queued');

      pollInterval = setInterval(() => pollJobStatus(jobId), 2000);
      pollJobStatus(jobId);
    }

    async function pollJobStatus(jobId) {
      if (!jobId || jobId !== currentJobId) return;
      const requestVersion = ++statusRequestVersion;
      try {
        const res = await fetch('/api/jobs/' + encodeURIComponent(jobId));
        if (jobId !== currentJobId || requestVersion !== statusRequestVersion) return;
        if (!res.ok) {
          if (res.status === 404) {
            stopPolling();
            updateStatusBadge('failed');
            jobErrorBox.textContent = 'Job not found on server.';
            jobErrorBox.classList.remove('hidden');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Analyze Video';
          }
          return;
        }

        const data = await res.json();
        if (jobId !== currentJobId || requestVersion !== statusRequestVersion) return;
        updateStatusBadge(data.status);

        if (data.status === 'completed' || data.status === 'incomplete') {
          stopPolling();
          submitBtn.disabled = false;
          submitBtn.textContent = 'Analyze Video';
          loadJobResults(jobId);
        } else if (data.status === 'failed' || data.status === 'interrupted') {
          stopPolling();
          submitBtn.disabled = false;
          submitBtn.textContent = 'Analyze Video';
          resultsCard.classList.add('hidden');
          const detail = data.error || (data.status === 'interrupted' ? 'Job was interrupted by server shutdown or restart.' : 'Job terminated with error.');
          jobErrorBox.textContent = detail;
          jobErrorBox.classList.remove('hidden');
        }
      } catch (err) {
        if (jobId !== currentJobId || requestVersion !== statusRequestVersion) return;
        // Network hiccups while polling: continue polling
      }
    }

    function stopPolling() {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    }

    function updateStatusBadge(status) {
      const norm = (status || 'unknown').toLowerCase();
      const label = norm.toUpperCase();
      statusBadge.className = 'status-badge status-' + norm;
      statusBadge.setAttribute('aria-label', 'Job status: ' + label);
      const statusEl = document.getElementById('job-status');
      if (statusEl) {
        statusEl.textContent = label;
      } else {
        statusBadge.textContent = label;
      }
    }

    async function loadJobResults(jobId) {
      if (!jobId || jobId !== currentJobId) return;
      const requestVersion = ++resultsRequestVersion;
      try {
        const res = await fetch('/api/jobs/' + encodeURIComponent(jobId) + '/results');
        if (jobId !== currentJobId || requestVersion !== resultsRequestVersion) return;
        if (!res.ok) throw new Error('Failed to load analysis results.');
        const data = await res.json();
        if (jobId !== currentJobId || requestVersion !== resultsRequestVersion) return;

        if (data.status === 'failed' || data.status === 'interrupted') {
          resultsCard.classList.add('hidden');
          const detail = data.error || (data.status === 'interrupted' ? 'Job was interrupted by server shutdown or restart.' : 'Job terminated with error.');
          jobErrorBox.textContent = detail;
          jobErrorBox.classList.remove('hidden');
          return;
        }

        displayResults(data);
        if (jobId !== currentJobId || requestVersion !== resultsRequestVersion) return;
        resultsCard.classList.remove('hidden');
        const resultsLive = document.getElementById('results-announcement');
        if (resultsLive) {
          if (data.status === 'incomplete') {
            resultsLive.textContent = 'Analysis ended early (incomplete video). Partial decoded-prefix results loaded below.';
          } else {
            resultsLive.textContent = 'Analysis complete. Results and movement observations loaded below.';
          }
        }
      } catch (err) {
        if (jobId !== currentJobId || requestVersion !== resultsRequestVersion) return;
        jobErrorBox.textContent = err.message || 'Error rendering results.';
        jobErrorBox.classList.remove('hidden');
        resultsCard.classList.add('hidden');
      }
    }

    function renderResults(data) {
      if (!data) return;
      clearPendingSeek();

      if (data.status === 'failed' || data.status === 'interrupted') {
        resultsCard.classList.add('hidden');
        if (jobErrorBox) {
          jobErrorBox.textContent = data.error || (data.status === 'interrupted' ? 'Job was interrupted by server shutdown or restart.' : 'Job terminated with error.');
          jobErrorBox.classList.remove('hidden');
        }
        return;
      }

      const repsVal = data.completed_reps != null ? String(data.completed_reps) : '0';
      metricReps.textContent = repsVal;
      metricFrames.textContent = data.processed_frames != null ? String(data.processed_frames) : '0';

      const attemptsList = (data.attempts && data.attempts.events) ? data.attempts.events : [];
      metricAttempts.textContent = String(attemptsList.length);

      const uncountedCount = attemptsList.filter(e => e.status === 'uncounted').length;
      metricUncounted.textContent = String(uncountedCount);

      const repsLabel = document.getElementById('metric-reps-label');
      const resultsTitle = document.getElementById('results-title');

      if (data.status === 'incomplete') {
        let declaredFrames = (data.summary && data.summary.source_video && data.summary.source_video.declared_frame_count != null) ? data.summary.source_video.declared_frame_count : null;
        let procFrames = data.processed_frames != null ? data.processed_frames : (data.summary && data.summary.processed_frames != null ? data.summary.processed_frames : 0);
        if (declaredFrames == null && data.read_status_note) {
          const match = data.read_status_note.match(/([0-9]+)[ \t]+decoded[ \t]+vs[ \t]+([0-9]+)[ \t]+declared/i) || data.read_status_note.match(/declared[ \t]+([0-9]+)[ \t]+frames.*?([0-9]+)[ \t]+frames[ \t]+were[ \t]+successfully[ \t]+decoded/i);
          if (match) {
            declaredFrames = match[2] || match[1];
            if (!procFrames) procFrames = match[1] || match[2];
          }
        }
        const frameDetail = declaredFrames ? (procFrames + ' of ' + declaredFrames + ' declared frames processed') : (procFrames + ' frames processed');
        if (repsLabel) {
          repsLabel.textContent = 'Decoded Prefix Repetitions: ' + repsVal + ' (Incomplete Video, ' + frameDetail + ')';
        }
        if (resultsTitle) {
          resultsTitle.textContent = 'Analysis Results (Incomplete Video - Decoded Video Prefix Only)';
        }
        const noteDetail = data.read_status_note || 'Processing ended before declared video duration.';
        incompleteNoteText.textContent = 'This count reflects a decoded prefix only (' + frameDetail + '), not a full completed video session. ' + noteDetail;
        incompleteNoteBox.classList.remove('hidden');
      } else {
        if (repsLabel) {
          repsLabel.textContent = 'Completed Repetitions';
        }
        if (resultsTitle) {
          resultsTitle.textContent = 'Analysis Results';
        }
        incompleteNoteBox.classList.add('hidden');
      }

      const phaseFeedback = data.phase_feedback || (data.summary && data.summary.phase_feedback) || null;
      const phaseFeedbackMap = {};
      if (phaseFeedback && Array.isArray(phaseFeedback.attempts)) {
        phaseFeedback.attempts.forEach(pa => {
          if (pa && pa.attempt_id != null) {
            phaseFeedbackMap[pa.attempt_id] = pa;
          }
        });
      }

      const feedback = data.feedback_observations || {};
      const feedbackSummary = feedback.summary || {};
      const sessionContainer = document.getElementById('session-observations');
      sessionContainer.textContent = '';
      const sessionTexts = [...(feedbackSummary.session_observations || [])];
      (feedbackSummary.comparisons || []).forEach(group => {
        sessionTexts.push(...(group.observations || []), ...(group.limitations || []));
      });
      if (phaseFeedback && phaseFeedback.summary && Array.isArray(phaseFeedback.summary.comparisons)) {
        phaseFeedback.summary.comparisons.forEach(comp => {
          if (comp && comp.observation && !sessionTexts.includes(comp.observation)) {
            sessionTexts.push(comp.observation);
          }
        });
      }
      if (!sessionTexts.length) sessionTexts.push('No within-session comparison is available for this analysis.');
      sessionTexts.forEach(text => {
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        sessionContainer.appendChild(paragraph);
      });
      if (phaseFeedback && Array.isArray(phaseFeedback.biomechanical_limitations) && phaseFeedback.biomechanical_limitations.length > 0) {
        const limBox = document.createElement('div');
        limBox.style.marginTop = '8px';
        limBox.style.fontSize = '0.8rem';
        limBox.style.color = 'var(--text-muted)';
        const limTitle = document.createElement('div');
        limTitle.style.fontWeight = '600';
        limTitle.textContent = 'Biomechanical Analysis Limitations:';
        limBox.appendChild(limTitle);
        phaseFeedback.biomechanical_limitations.forEach(limText => {
          const lp = document.createElement('div');
          lp.textContent = '• ' + limText;
          limBox.appendChild(lp);
        });
        sessionContainer.appendChild(limBox);
      }
      const feedbackMap = {};
      (feedback.attempts || []).forEach(event => { feedbackMap[event.attempt_id] = event; });

      // Render attempts table
      attemptsTbody.textContent = '';
      if (attemptsList.length === 0) {
        const row = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 6;
        td.style.textAlign = 'center';
        td.textContent = 'No discrete movement cycles detected in the input video.';
        row.appendChild(td);
        attemptsTbody.appendChild(row);
      } else {
        const postureEvents = (data.posture_summary && data.posture_summary.events) ? data.posture_summary.events : [];
        const postureMap = {};
        postureEvents.forEach(pe => {
          if (pe.attempt_id != null) postureMap[pe.attempt_id] = pe;
        });

        attemptsList.forEach(att => {
          const row = document.createElement('tr');

          const tdId = document.createElement('td');
          tdId.textContent = '#' + (att.attempt_id != null ? att.attempt_id : '-');
          row.appendChild(tdId);

          const tdStatus = document.createElement('td');
          const spanStatus = document.createElement('span');
          spanStatus.textContent = (att.status || 'uncounted').toUpperCase();
          spanStatus.className = 'status-badge ' + (att.status === 'counted' ? 'status-completed' : 'status-incomplete');
          tdStatus.appendChild(spanStatus);
          row.appendChild(tdStatus);

          const tdReason = document.createElement('td');
          tdReason.textContent = (att.reason || 'completed_cycle').replace(/_/g, ' ');
          row.appendChild(tdReason);

          const tdFrames = document.createElement('td');
          const startF = att.start_frame != null ? att.start_frame : 0;
          const endF = att.end_frame != null ? att.end_frame : 0;
          const startT = att.start_time_s != null ? att.start_time_s.toFixed(2) : '0.00';
          const endT = att.end_time_s != null ? att.end_time_s.toFixed(2) : '0.00';
          tdFrames.textContent = startF + '-' + endF + ' (' + startT + 's-' + endT + 's)';
          row.appendChild(tdFrames);

          const tdKnee = document.createElement('td');
          tdKnee.textContent = (att.min_knee_angle_deg != null) ? att.min_knee_angle_deg.toFixed(1) + '°' : 'N/A';
          row.appendChild(tdKnee);

          const tdExpl = document.createElement('td');
          const explDiv = document.createElement('div');
          explDiv.textContent = att.explanation || 'Cycle completed within threshold criteria.';
          tdExpl.appendChild(explDiv);

          // Add posture observations if available
          const observedFeedback = feedbackMap[att.attempt_id];
          if (observedFeedback && observedFeedback.movement_explanation) {
            const context = document.createElement('p');
            context.textContent = observedFeedback.movement_explanation.context_note || '';
            tdExpl.appendChild(context);
          }
          const post = observedFeedback || postureMap[att.attempt_id];
          if (post && post.observations && post.observations.length > 0) {
            const postDiv = document.createElement('div');
            postDiv.style.marginTop = '4px';
            postDiv.style.fontSize = '0.8rem';
            postDiv.style.color = 'var(--text-muted)';
            post.observations.forEach(obsText => {
              const p = document.createElement('div');
              p.textContent = '• ' + obsText;
              postDiv.appendChild(p);
            });
            tdExpl.appendChild(postDiv);
          }

          // Phase-localized review cues and seek actions
          const phaseAtt = phaseFeedbackMap[att.attempt_id];
          if (phaseAtt) {
            const phaseBlock = document.createElement('div');
            phaseBlock.className = 'phase-feedback-block';
            phaseBlock.style.marginTop = '6px';
            phaseBlock.style.paddingTop = '6px';
            phaseBlock.style.borderTop = '1px dashed var(--card-border)';

            if (phaseAtt.phase_analysis_available === true) {
              const cues = Array.isArray(phaseAtt.review_cues) ? phaseAtt.review_cues : [];
              if (cues.length > 0) {
                const cuesDiv = document.createElement('div');
                cuesDiv.style.fontSize = '0.82rem';
                cues.forEach(cueText => {
                  const cp = document.createElement('div');
                  cp.textContent = '• ' + cueText;
                  cuesDiv.appendChild(cp);
                });
                phaseBlock.appendChild(cuesDiv);
              }

              const btnContainer = document.createElement('div');
              btnContainer.style.display = 'flex';
              btnContainer.style.flexWrap = 'wrap';
              btnContainer.style.gap = '6px';
              btnContainer.style.marginTop = '6px';
              btnContainer.style.alignItems = 'center';

              let hasTimestampAction = false;

              if (phaseAtt.min_knee_point) {
                const tKnee = phaseAtt.min_knee_point.video_time_s;
                if (isValidTimestamp(tKnee)) {
                  hasTimestampAction = true;
                  const btn = document.createElement('button');
                  btn.type = 'button';
                  btn.className = 'btn-seek';
                  btn.textContent = 'Seek turnaround (' + Number(tKnee).toFixed(2) + 's)';
                  btn.title = 'Seek video to turnaround candidate point';
                  btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    seekVideoToTime(tKnee);
                  });
                  btnContainer.appendChild(btn);
                } else {
                  hasTimestampAction = true;
                  const noTime = document.createElement('span');
                  noTime.style.fontSize = '0.78rem';
                  noTime.style.color = 'var(--text-muted)';
                  noTime.textContent = 'Turnaround: no timestamp';
                  btnContainer.appendChild(noTime);
                }
              }

              if (phaseAtt.return_phase) {
                const tRet = phaseAtt.return_phase.delta_start_time_s;
                if (isValidTimestamp(tRet)) {
                  hasTimestampAction = true;
                  const btn = document.createElement('button');
                  btn.type = 'button';
                  btn.className = 'btn-seek';
                  btn.textContent = 'Seek return (' + Number(tRet).toFixed(2) + 's)';
                  btn.title = 'Seek video to return phase start';
                  btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    seekVideoToTime(tRet);
                  });
                  btnContainer.appendChild(btn);
                } else if (phaseAtt.return_phase.delta_start_frame != null) {
                  hasTimestampAction = true;
                  const noTime = document.createElement('span');
                  noTime.style.fontSize = '0.78rem';
                  noTime.style.color = 'var(--text-muted)';
                  noTime.textContent = 'Return phase: no timestamp';
                  btnContainer.appendChild(noTime);
                }
              }

              if (hasTimestampAction) {
                phaseBlock.appendChild(btnContainer);
              }
            } else {
              const suppDiv = document.createElement('div');
              suppDiv.style.fontSize = '0.8rem';
              suppDiv.style.color = 'var(--text-muted)';
              const rawReason = (phaseAtt.suppression_reason || 'suppressed').replace(/_/g, ' ');
              let suppReason = rawReason;
              if (!rawReason.toLowerCase().startsWith('suppressed')) {
                suppReason = 'suppressed due to ' + rawReason;
              }
              suppDiv.textContent = 'Phase review: ' + suppReason;
              phaseBlock.appendChild(suppDiv);
            }

            tdExpl.appendChild(phaseBlock);
          }

          row.appendChild(tdExpl);
          attemptsTbody.appendChild(row);
        });
      }

      // Visual artifacts
      const plotUrl = '/api/jobs/' + encodeURIComponent(data.job_id) + '/download/angle_plot.png';
      anglePlotImg.src = plotUrl;

      const preview = data.available_artifacts && data.available_artifacts['preview.mp4'];
      const videoName = preview && preview.available ? 'preview.mp4' : 'annotated.mp4';
      document.getElementById('video-fallback-notice').classList.toggle('hidden', !!(preview && preview.available));
      const videoUrl = '/api/jobs/' + encodeURIComponent(data.job_id) + '/download/' + videoName;
      videoSource.src = videoUrl;
      annotatedVideo.load();

      // Download buttons
      downloadsContainer.textContent = '';
      if (data.available_artifacts) {
        ALLOWED_ARTIFACT_NAMES.forEach(name => {
          const item = data.available_artifacts[name];
          if (item && item.available) {
            const a = document.createElement('a');
            a.href = item.download_url || ('/api/jobs/' + encodeURIComponent(data.job_id) + '/download/' + encodeURIComponent(name));
            a.className = 'btn-secondary';
            a.textContent = 'Download ' + name + (item.size_bytes ? ' (' + (item.size_bytes / 1024).toFixed(0) + ' KB)' : '');
            a.setAttribute('download', name);
            a.setAttribute('role', 'button');
            downloadsContainer.appendChild(a);
          }
        });
      }

      renderCoachingFeedback(data.coaching_feedback);
    }

    function displayResults(data) {
      renderResults(data);
    }

    function renderCoachingFeedback(coaching) {
      const coachingCard = document.getElementById('coaching-feedback-card');
      const coachingStatusBadge = document.getElementById('coaching-status-badge');
      const coachingClinicalDisclaimer = document.getElementById('coaching-clinical-disclaimer');
      const coachingValidationStatus = document.getElementById('coaching-validation-status');
      const coachingQualityDetails = document.getElementById('coaching-quality-details');
      const coachingCuesList = document.getElementById('coaching-cues-list');
      const coachingUncountedList = document.getElementById('coaching-uncounted-list');

      if (!coachingCard) return;

      const isUnavailable = !coaching ||
        (typeof coaching === 'object' && Object.keys(coaching).length === 0) ||
        (coaching.status && String(coaching.status).toUpperCase() === 'UNAVAILABLE');

      if (isUnavailable) {
        if (coachingStatusBadge) {
          coachingStatusBadge.textContent = 'UNAVAILABLE';
          coachingStatusBadge.style.background = '#475569';
        }
        if (coachingQualityDetails) coachingQualityDetails.textContent = 'Coaching telemetry is not available for this run.';
        if (coachingCuesList) coachingCuesList.textContent = '';
        if (coachingUncountedList) coachingUncountedList.textContent = '';
        if (coachingClinicalDisclaimer && coaching && coaching.disclaimers && coaching.disclaimers.clinical_disclaimer) {
          coachingClinicalDisclaimer.textContent = coaching.disclaimers.clinical_disclaimer;
        }
        if (coachingValidationStatus && coaching && coaching.disclaimers && coaching.disclaimers.human_validation_status) {
          coachingValidationStatus.textContent = coaching.disclaimers.human_validation_status;
        }
        return;
      }

      if (coachingStatusBadge) {
        const rawStatus = (coaching.status || 'UNVALIDATED').toUpperCase();
        coachingStatusBadge.textContent = rawStatus;
      }
      if (coachingClinicalDisclaimer && coaching.disclaimers && coaching.disclaimers.clinical_disclaimer) {
        coachingClinicalDisclaimer.textContent = coaching.disclaimers.clinical_disclaimer;
      }
      if (coachingValidationStatus && coaching.disclaimers && coaching.disclaimers.human_validation_status) {
        coachingValidationStatus.textContent = coaching.disclaimers.human_validation_status;
      }

      if (coachingQualityDetails) {
        coachingQualityDetails.textContent = '';
        const recQuality = (coaching.overall_summary && coaching.overall_summary.recording_quality) || {};
        let viewText = 'Unmeasured (2D lateral view)';
        if (recQuality.camera_view_adequate === true) {
          viewText = 'Adequate';
        } else if (recQuality.camera_view_adequate === false) {
          viewText = 'Suboptimal';
        } else if (recQuality.camera_view_adequate === null || recQuality.camera_view_adequate === undefined) {
          viewText = 'Unmeasured (2D lateral view)';
        }
        const contText = recQuality.tracking_continuity || 'Not assessed';
        const covText = recQuality.posture_coverage_pct != null ? (Number(recQuality.posture_coverage_pct).toFixed(1) + '%') : 'N/A';

        const summaryLine = document.createElement('div');
        summaryLine.textContent = 'Camera view geometry: ' + viewText + ' | Tracking continuity: ' + contText + ' | Tracking coverage: ' + covText;
        coachingQualityDetails.appendChild(summaryLine);

        const distinctionNote = document.createElement('div');
        distinctionNote.style.fontSize = '0.78rem';
        distinctionNote.style.color = 'var(--text-muted)';
        distinctionNote.style.marginTop = '2px';
        distinctionNote.textContent = 'Note: 2D tracking coverage measures monocular joint visibility, not 3D camera orientation geometry.';
        coachingQualityDetails.appendChild(distinctionNote);

        if (Array.isArray(recQuality.notes) && recQuality.notes.length > 0) {
          const notesDiv = document.createElement('div');
          notesDiv.style.marginTop = '4px';
          recQuality.notes.forEach(note => {
            const noteItem = document.createElement('div');
            noteItem.textContent = '• ' + note;
            notesDiv.appendChild(noteItem);
          });
          coachingQualityDetails.appendChild(notesDiv);
        }
      }

      if (coachingCuesList) {
        coachingCuesList.textContent = '';
        const cues = Array.isArray(coaching.per_repetition_cues) ? coaching.per_repetition_cues : [];
        if (cues.length === 0) {
          const noCues = document.createElement('div');
          noCues.style.fontSize = '0.82rem';
          noCues.style.color = 'var(--text-muted)';
          noCues.textContent = 'No repetition cues generated.';
          coachingCuesList.appendChild(noCues);
        } else {
          cues.forEach(cue => {
            const cueItem = document.createElement('div');
            cueItem.style.background = '#1e293b';
            cueItem.style.padding = '8px 12px';
            cueItem.style.borderRadius = '4px';
            cueItem.style.border = '1px solid var(--card-border)';
            cueItem.style.fontSize = '0.82rem';

            const header = document.createElement('div');
            header.style.fontWeight = '600';
            header.style.color = 'var(--primary)';
            const repLabel = 'Repetition #' + (cue.rep_index != null ? cue.rep_index : (cue.attempt_id != null ? cue.attempt_id : '-'));
            const countBadge = cue.is_counted ? ' [Counted]' : ' [Uncounted]';
            header.textContent = repLabel + countBadge;
            cueItem.appendChild(header);

            if (cue.measurements) {
              const m = cue.measurements;
              const mParts = [];
              if (m.min_knee_angle_deg != null) mParts.push('Min knee: ' + Number(m.min_knee_angle_deg).toFixed(1) + '°');
              if (m.median_torso_inclination_deg != null) mParts.push('Torso inclination: ' + Number(m.median_torso_inclination_deg).toFixed(1) + '°');
              if (m.median_hip_angle_deg != null) mParts.push('Hip: ' + Number(m.median_hip_angle_deg).toFixed(1) + '°');
              if (m.coverage_pct != null) mParts.push('Coverage: ' + Number(m.coverage_pct).toFixed(0) + '%');
              if (mParts.length > 0) {
                const mDiv = document.createElement('div');
                mDiv.style.color = 'var(--text-muted)';
                mDiv.style.marginTop = '2px';
                mDiv.textContent = mParts.join(' | ');
                cueItem.appendChild(mDiv);
              }
            }

            if (Array.isArray(cue.observations) && cue.observations.length > 0) {
              const obsContainer = document.createElement('div');
              obsContainer.style.marginTop = '4px';
              cue.observations.forEach(obs => {
                const oDiv = document.createElement('div');
                const conf = obs.confidence_level ? (' [' + obs.confidence_level + ']') : '';
                oDiv.textContent = '• ' + (obs.message || obs.rule_id || '') + conf;
                obsContainer.appendChild(oDiv);
              });
              cueItem.appendChild(obsContainer);
            }

            if (Array.isArray(cue.suggestions) && cue.suggestions.length > 0) {
              const sugContainer = document.createElement('div');
              sugContainer.style.marginTop = '4px';
              sugContainer.style.color = '#38bdf8';
              cue.suggestions.forEach(sug => {
                const sDiv = document.createElement('div');
                sDiv.textContent = '→ Suggestion: ' + (sug.message || sug.rule_id || '');
                sugContainer.appendChild(sDiv);
              });
              cueItem.appendChild(sugContainer);
            }

            coachingCuesList.appendChild(cueItem);
          });
        }
      }

      if (coachingUncountedList) {
        coachingUncountedList.textContent = '';
        const uncounted = Array.isArray(coaching.uncounted_movement_explanations) ? coaching.uncounted_movement_explanations : [];
        if (uncounted.length === 0) {
          const noUncounted = document.createElement('div');
          noUncounted.style.fontSize = '0.82rem';
          noUncounted.style.color = 'var(--text-muted)';
          noUncounted.textContent = 'No uncounted movements detected.';
          coachingUncountedList.appendChild(noUncounted);
        } else {
          uncounted.forEach(item => {
            const uDiv = document.createElement('div');
            uDiv.style.background = '#1e293b';
            uDiv.style.padding = '6px 10px';
            uDiv.style.borderRadius = '4px';
            uDiv.style.fontSize = '0.82rem';
            uDiv.style.border = '1px solid var(--card-border)';

            const title = document.createElement('div');
            title.style.fontWeight = '600';
            title.textContent = 'Attempt #' + (item.attempt_id != null ? item.attempt_id : '-') + ' (' + (item.reason_code || 'uncounted') + ')';
            uDiv.appendChild(title);

            const expl = document.createElement('div');
            expl.textContent = item.explanation || '';
            uDiv.appendChild(expl);

            if (item.context_hypothesis) {
              const hyp = document.createElement('div');
              hyp.style.color = 'var(--warning)';
              hyp.style.marginTop = '2px';
              const hypMap = {
                'possible_setup_or_weight_return': 'Possible setup or weight return (non-repetition motion)',
                'shallow_depth': 'Shallow depth (inflection remained above configured threshold)',
                'tracking_lost': 'Tracking lost (confidence drop, occlusion, or tracking gap)',
                'unfinished_cycle': 'Unfinished cycle (repetition in progress or video ended early)',
                'ambiguous': 'Ambiguous context (multiple subjects or conflicting signals)',
                'unknown': 'Unknown context (unclassified movement event)'
              };
              const hypLabel = hypMap[item.context_hypothesis] || item.context_hypothesis.replace(/_/g, ' ');
              hyp.textContent = 'Context Hypothesis: ' + hypLabel;
              uDiv.appendChild(hyp);
            }
            coachingUncountedList.appendChild(uDiv);
          });
        }
      }
    }
    let restoredJob = null;
    try {
      if (typeof URLSearchParams !== 'undefined' && typeof window !== 'undefined' && window.location && window.location.search) {
        restoredJob = new URLSearchParams(window.location.search).get('job');
      }
    } catch (_) {}
    if (restoredJob && /^[0-9a-f-]{36}$/i.test(restoredJob)) {
      currentJobId = restoredJob;
      startJobTracking(restoredJob, 'Previously uploaded video');
    }

    try {
      const outerGlobal = (typeof console !== 'undefined' && console.log && console.log.constructor) ?
        console.log.constructor('return this')() : null;
      if (outerGlobal) {
        if (!outerGlobal.document && typeof document !== 'undefined') outerGlobal.document = document;
        if (!outerGlobal.window && typeof window !== 'undefined') outerGlobal.window = window;
        try {
          Object.defineProperty(outerGlobal, 'currentJobId', {
            get: () => currentJobId,
            set: (val) => { currentJobId = val; },
            configurable: true
          });
        } catch (_) { outerGlobal.currentJobId = currentJobId; }
        outerGlobal.getCurrentJobId = () => currentJobId;
        outerGlobal.setCurrentJobId = (val) => { currentJobId = val; };
        outerGlobal.renderResults = renderResults;
        outerGlobal.displayResults = displayResults;
        outerGlobal.renderCoachingFeedback = renderCoachingFeedback;
        outerGlobal.pollJobStatus = pollJobStatus;
        outerGlobal.loadJobResults = loadJobResults;
        outerGlobal.updateStatusBadge = updateStatusBadge;
        outerGlobal.startJobTracking = startJobTracking;
        outerGlobal.stopPolling = stopPolling;
        outerGlobal.clearPendingSeek = clearPendingSeek;
        outerGlobal.seekVideoToTime = seekVideoToTime;
        outerGlobal.validateClientFile = validateClientFile;
      }
    } catch (_) {}

    if (typeof module !== 'undefined' && module.exports) {
      module.exports = {
        MAX_UPLOAD_BYTES,
        ALLOWED_EXTENSIONS,
        ALLOWED_ARTIFACT_NAMES,
        formatFileSize,
        getFileExtension,
        validateClientFile,
        isValidTimestamp,
        clearPendingSeek,
        seekVideoToTime,
        updateStatusBadge,
        startJobTracking,
        pollJobStatus,
        stopPolling,
        loadJobResults,
        renderResults,
        displayResults,
        renderCoachingFeedback,
        getCurrentJobId: () => currentJobId,
        setCurrentJobId: (id) => { currentJobId = id; },
      };
    }
  </script>
</body>
</html>
"""


def create_app(
    job_manager: Optional[JobManager] = None,
    *,
    enable_review: bool = True,
    review_storage_root: Optional[Union[str, Path]] = None,
    review_max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    **kwargs: Any,
) -> Any:
    """FastAPI application factory for 2D Gym Pose Analyser.

    Allows injecting a configured JobManager for isolated unit and integration testing,
    and optionally mounts the raw human video review workspace under /review.
    """
    if kwargs:
        raise TypeError(f"Unexpected keyword argument: {list(kwargs.keys())[0]}")

    if FastAPI is None:
        raise ImportError(
            "FastAPI is not installed. Please install optional dependencies from gym_pose/requirements-web.txt."
        )

    # Initialize default local JobManager if not provided
    if job_manager is None:
        gym_pose_dir = Path(__file__).resolve().parent.parent
        storage_dir = gym_pose_dir / "output" / "web_jobs"
        model_path = gym_pose_dir / "models" / "yolo11n-pose.pt"
        job_manager = JobManager(
            storage_dir=storage_dir,
            device="cpu",
            model_path=model_path,
            gym_pose_dir=gym_pose_dir,
            max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES,
        )

    app = FastAPI(
        title="2D Gym Pose Analyser",
        description="Local web interface for single-user squat video evaluation and kinematic feedback.",
        version="1.0.0",
    )

    if not isinstance(enable_review, bool):
        raise TypeError("enable_review must be a boolean")
    if enable_review:
        from .review_web import create_review_app
        from .review_store import DEFAULT_STORAGE_ROOT

        review_root_path = Path(review_storage_root if review_storage_root is not None else DEFAULT_STORAGE_ROOT)
        review_root = review_root_path.resolve()
        jobs_root = Path(job_manager.storage_dir).resolve()
        if review_root == jobs_root or review_root in jobs_root.parents or jobs_root in review_root.parents:
            raise ValueError("Human-review and analysis-job storage must be separate directories")
        app.mount("/review", create_review_app(
            storage_root=review_root_path,
            max_upload_bytes=review_max_upload_bytes,
        ))

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the self-contained single-page web interface."""
        html = INDEX_HTML
        if not enable_review:
            import re
            html = re.sub(r'<nav class="review-nav".*?</nav>', "", html, count=1, flags=re.DOTALL)
            html = re.sub(r'<div><strong>Human Review:</strong>.*?</div>', "", html, count=1, flags=re.DOTALL)
        return HTMLResponse(content=html, status_code=200)

    @app.get("/api/status")
    async def get_server_status() -> Dict[str, Any]:
        """Return worker busy state and ID of active job if running."""
        return {
            "busy": job_manager.is_busy(),
            "active_job_id": job_manager.get_active_job_id(),
        }

    @app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> JSONResponse:
        """Upload video file and enqueue bounded single-worker analysis job.

        Returns HTTP 429 if server worker is currently active on another job.
        Enforces maximum upload size, chunked transfer, and clean rollback on error.
        """
        if job_manager.is_busy():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Server is busy: an analysis job is currently active. One analysis at a time.",
            )

        try:
            job_id, _target_path, total_bytes = job_manager.save_upload_file(
                file_obj=file.file,
                raw_filename=file.filename or "video.mp4",
            )
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save upload: {exc}",
            )
        finally:
            await file.close()

        try:
            job_manager.enqueue_job(job_id, background_tasks=background_tasks)
        except JobBusyError as jbe:
            try:
                import shutil
                shutil.rmtree(job_manager._get_job_dir(job_id), ignore_errors=True)
            except Exception:
                pass
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(jbe))

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "status": "queued",
                "filename": file.filename or "video.mp4",
                "file_size_bytes": total_bytes,
            },
        )

    @app.get("/api/jobs/{job_id}")
    async def get_job_status(job_id: str) -> Dict[str, Any]:
        """Return lifecycle status metadata for a specific job ID."""
        if not is_valid_job_id(job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid job ID format. Standard UUID4 required.",
            )
        meta = job_manager.get_metadata(job_id)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        return meta

    @app.get("/api/jobs/{job_id}/results")
    async def get_job_results_endpoint(job_id: str) -> Dict[str, Any]:
        """Return aggregated reviewed JSON outputs and artifact links for a job."""
        if not is_valid_job_id(job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid job ID format. Standard UUID4 required.",
            )
        try:
            return job_manager.get_job_results(job_id)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load job results: {exc}",
            )

    @app.get("/api/jobs/{job_id}/download/{artifact_name}")
    async def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
        """Download a verified analysis artifact from the strict allowlist."""
        if not is_valid_job_id(job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid job ID format. Standard UUID4 required.",
            )
        if (
            artifact_name not in ALLOWED_ARTIFACTS
            or "/" in artifact_name
            or "\\" in artifact_name
            or ".." in artifact_name
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact '{artifact_name}' is not in the allowed export list.",
            )
        try:
            artifact_path = job_manager.get_artifact_path(job_id, artifact_name)
            media_type = ALLOWED_ARTIFACTS.get(artifact_name, "application/octet-stream")
            disposition = "inline" if media_type.startswith(("video/", "image/")) else "attachment"
            return FileResponse(
                path=str(artifact_path),
                media_type=media_type,
                filename=artifact_name,
                headers={"Content-Disposition": f'{disposition}; filename="{artifact_name}"'},
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact '{artifact_name}' does not exist for job '{job_id}'.",
            )
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    return app


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for local web server launcher."""
    parser = argparse.ArgumentParser(
        prog="python -m gym_analyser.web",
        description="2D Gym Pose Analyser: local single-user web interface launcher.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Loopback interface host (default: 127.0.0.1). LAN binding is restricted.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference compute device for child analysis worker (e.g. 'cpu' or '0').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n-pose.pt",
        help="YOLO pose model path relative to gym_pose (default: models/yolo11n-pose.pt).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-job analysis execution timeout in seconds (default: 600.0).",
    )
    return parser


def main() -> None:
    """CLI launcher entry point for the local web server."""
    parser = build_parser()
    args = parser.parse_args()

    # Loopback restriction check
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        sys.stderr.write(
            f"Security Error: Host '{args.host}' is not a local loopback interface. "
            "Only 127.0.0.1, localhost, or ::1 is permitted.\n"
        )
        sys.exit(1)

    gym_pose_dir = Path(__file__).resolve().parent.parent
    model_path = (gym_pose_dir / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    storage_dir = gym_pose_dir / "output" / "web_jobs"

    job_manager = JobManager(
        storage_dir=storage_dir,
        device=args.device,
        model_path=model_path,
        gym_pose_dir=gym_pose_dir,
        timeout_seconds=args.timeout,
    )

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "Error: uvicorn is not installed. Please install optional dependencies: "
            "pip install -r requirements-web.txt\n"
        )
        sys.exit(1)

    app = create_app(job_manager)
    print(f"Starting 2D Gym Pose Analyser Web UI on http://{args.host}:{args.port}")
    print(f"Human Review Workspace available on http://{args.host}:{args.port}/review/")
    print(f"Compute device for analysis child process: {args.device}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
