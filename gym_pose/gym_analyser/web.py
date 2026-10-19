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
from typing import Any, Dict, Optional

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

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
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
    .container { max-width: 980px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
    header { text-align: center; margin-bottom: 8px; }
    header h1 { font-size: 1.85rem; font-weight: 700; color: var(--primary); }
    header p { color: var(--text-muted); font-size: 0.95rem; margin-top: 4px; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 20px;
    }
    .card h2 { font-size: 1.15rem; font-weight: 600; margin-bottom: 12px; color: var(--text); }
    .notice-box {
      background: #0f172a;
      border-left: 4px solid var(--primary);
      padding: 12px 16px;
      border-radius: 4px;
      font-size: 0.88rem;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .notice-box strong { color: var(--text); }
    .upload-form { display: flex; flex-direction: column; gap: 14px; }
    .file-input-wrapper {
      border: 2px dashed var(--card-border);
      border-radius: 6px;
      padding: 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s;
    }
    .file-input-wrapper:hover { border-color: var(--primary); }
    input[type="file"] { display: none; }
    .file-label { cursor: pointer; color: var(--primary); font-weight: 500; font-size: 0.95rem; }
    .selected-filename { margin-top: 8px; font-size: 0.85rem; color: var(--text-muted); }
    .btn {
      background: var(--primary);
      color: #0f172a;
      border: none;
      padding: 10px 18px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      transition: background 0.2s, opacity 0.2s;
      align-self: flex-start;
    }
    .btn:hover:not(:disabled) { background: var(--primary-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-secondary {
      background: #334155;
      color: var(--text);
      text-decoration: none;
      font-size: 0.82rem;
      padding: 6px 12px;
      border-radius: 4px;
      display: inline-block;
    }
    .btn-secondary:hover { background: #475569; }
    .status-badge {
      display: inline-block;
      padding: 4px 10px;
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
    .status-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
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
      padding: 12px 16px;
      text-align: center;
    }
    .metric-value { font-size: 1.75rem; font-weight: 700; color: var(--primary); }
    .metric-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      margin-top: 12px;
    }
    .data-table th, .data-table td {
      border: 1px solid var(--card-border);
      padding: 8px 12px;
      text-align: left;
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
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .media-box img, .media-box video {
      width: 100%;
      max-height: 480px;
      object-fit: contain;
      border-radius: 4px;
      background: #000;
    }
    .media-caption { font-size: 0.82rem; color: var(--text-muted); }
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
    }
    .alert-danger { background: #450a0a; border: 1px solid #7f1d1d; color: #fca5a5; }
    .alert-warning { background: #451a03; border: 1px solid #78350f; color: #fde68a; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>2D Gym Pose Analyser</h1>
      <p>Single-user local squat repetition counter and 2D kinematic feedback</p>
    </header>

    <!-- Recording Guidance & Technical Notices -->
    <div class="card">
      <h2>Recording Guidance & Operational Constraints</h2>
      <div class="notice-box">
        <div><strong>Camera Setup:</strong> Single-person, stationary camera at waist height perpendicular to movement plane, showing full body from head to feet in side view.</div>
        <div><strong>Kinematic Feedback:</strong> Measurements represent projected 2D coordinates from this single camera view. They do not constitute clinical safety evaluations or medical form judgements.</div>
        <div><strong>Uncounted Movements:</strong> Movements not meeting threshold criteria (such as stance setup, weight pickup, or weight return) are reported as uncounted attempts. An uncounted event does not indicate bad form.</div>
      </div>
    </div>

    <!-- Upload Form -->
    <div class="card" id="upload-card">
      <h2>Upload Squat Video</h2>
      <form class="upload-form" id="upload-form">
        <div class="file-input-wrapper" id="drop-zone" onclick="document.getElementById('video-input').click()">
          <input type="file" id="video-input" name="file" accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v">
          <div class="file-label">Click to select video file (or drag & drop here)</div>
          <div class="selected-filename" id="file-name-display">No file selected (Max 200MB: MP4, MOV, AVI, MKV, WebM)</div>
        </div>
        <button type="submit" class="btn" id="submit-btn" disabled>Analyze Video</button>
      </form>
      <div id="upload-error" class="alert-box alert-danger hidden"></div>
    </div>

    <!-- Active Job Status Panel -->
    <div class="card hidden" id="status-card">
      <div class="status-header">
        <h2>Analysis Job Status</h2>
        <span class="status-badge" id="status-badge">QUEUED</span>
      </div>
      <div><strong>Job ID:</strong> <span id="job-id-display"></span></div>
      <div><strong>File:</strong> <span id="job-filename-display"></span></div>
      <div id="job-error-box" class="alert-box alert-danger hidden"></div>
      <div id="job-warning-box" class="alert-box alert-warning hidden"></div>
    </div>

    <!-- Analysis Results Section -->
    <div class="card hidden" id="results-card">
      <h2>Analysis Results</h2>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value" id="metric-reps">0</div>
          <div class="metric-label">Completed Repetitions</div>
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
      <div id="incomplete-note-box" class="alert-box alert-warning hidden">
        <strong>Incomplete Processing Warning:</strong> <span id="incomplete-note-text"></span>
      </div>

      <div style="margin-top: 16px;">
        <h3>Movement observations</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted);">Comparisons describe measured counted cycles in this video. They do not grade technique or establish safety.</p>
        <div id="session-observations"></div>
      </div>

      <!-- Attempt Observations Table -->
      <div style="margin-top: 16px;">
        <h3>Observed Movement Events</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">
          Includes counted repetitions and uncounted events (e.g. setup, depth threshold not reached, tracking interruption).
        </p>
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

      <!-- Visual Media: Angle Plot and Annotated Video -->
      <div class="media-container">
        <div class="media-box" id="plot-box">
          <h3>Knee Flexion Angle Time Series</h3>
          <img id="angle-plot-img" src="" alt="Knee Angle Time Series Plot">
          <div class="media-caption">Knee angle trajectory with configured up (extension) and down (inflection) thresholds.</div>
        </div>

        <div class="media-box" id="video-box">
          <h3>Annotated Video Output</h3>
          <video id="annotated-video" controls preload="metadata">
            <source id="video-source" src="" type="video/mp4">
            Your browser does not support HTML5 video playback.
          </video>
          <div id="video-fallback-notice" class="notice-box" style="margin-top: 8px;">
            <strong>Browser Playback Notice:</strong> Video is rendered in OpenCV standard 'mp4v' format. Some web browsers require H.264 video streams and may display a blank player. Use the button below to download the annotated video file for local playback in VLC or QuickTime.
          </div>
        </div>
      </div>

      <!-- Downloadable Artifacts -->
      <div style="margin-top: 20px;">
        <h3>Exported Artifacts</h3>
        <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">Download raw telemetry and verified analysis outputs:</p>
        <div class="downloads-grid" id="downloads-container"></div>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('video-input');
    const fileNameDisplay = document.getElementById('file-name-display');
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

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        fileNameDisplay.textContent = fileInput.files[0].name + ' (' + (fileInput.files[0].size / (1024 * 1024)).toFixed(2) + ' MB)';
        submitBtn.disabled = false;
      } else {
        fileNameDisplay.textContent = 'No file selected (Max 200MB)';
        submitBtn.disabled = true;
      }
    });

    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!fileInput.files || fileInput.files.length === 0) return;

      submitBtn.disabled = true;
      uploadError.classList.add('hidden');
      uploadError.textContent = '';

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      try {
        const res = await fetch('/api/jobs', {
          method: 'POST',
          body: formData
        });

        if (res.status === 429) {
          const data = await res.json();
          uploadError.textContent = data.detail || 'Server is busy processing another analysis job. Please wait.';
          uploadError.classList.remove('hidden');
          submitBtn.disabled = false;
          return;
        }

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: 'Upload failed' }));
          throw new Error(errData.detail || 'Upload failed with status ' + res.status);
        }

        const data = await res.json();
        currentJobId = data.job_id;
        history.replaceState(null, "", "?job=" + encodeURIComponent(currentJobId));
        startJobTracking(currentJobId, data.filename);
      } catch (err) {
        uploadError.textContent = err.message || 'Network error during upload.';
        uploadError.classList.remove('hidden');
        submitBtn.disabled = false;
      }
    });

    function startJobTracking(jobId, filename) {
      statusCard.classList.remove('hidden');
      resultsCard.classList.add('hidden');
      jobIdDisplay.textContent = jobId;
      jobFilenameDisplay.textContent = filename || 'Video';
      jobErrorBox.classList.add('hidden');
      jobWarningBox.classList.add('hidden');

      updateStatusBadge('queued');

      if (pollInterval) clearInterval(pollInterval);
      pollInterval = setInterval(() => pollJobStatus(jobId), 2000);
      pollJobStatus(jobId);
    }

    async function pollJobStatus(jobId) {
      try {
        const res = await fetch('/api/jobs/' + encodeURIComponent(jobId));
        if (!res.ok) {
          if (res.status === 404) {
            stopPolling();
            updateStatusBadge('failed');
            jobErrorBox.textContent = 'Job not found on server.';
            jobErrorBox.classList.remove('hidden');
            submitBtn.disabled = false;
          }
          return;
        }

        const data = await res.json();
        updateStatusBadge(data.status);

        if (data.status === 'completed' || data.status === 'incomplete') {
          stopPolling();
          submitBtn.disabled = false;
          loadJobResults(jobId);
        } else if (data.status === 'failed' || data.status === 'interrupted') {
          stopPolling();
          submitBtn.disabled = false;
          jobErrorBox.textContent = data.error || 'Job terminated with error.';
          jobErrorBox.classList.remove('hidden');
        }
      } catch (err) {
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
      statusBadge.textContent = (status || 'unknown').toUpperCase();
      statusBadge.className = 'status-badge status-' + (status || 'queued');
    }

    async function loadJobResults(jobId) {
      try {
        const res = await fetch('/api/jobs/' + encodeURIComponent(jobId) + '/results');
        if (!res.ok) throw new Error('Failed to load analysis results.');
        const data = await res.json();

        renderResults(data);
        resultsCard.classList.remove('hidden');
      } catch (err) {
        jobErrorBox.textContent = err.message || 'Error rendering results.';
        jobErrorBox.classList.remove('hidden');
      }
    }

    function renderResults(data) {
      metricReps.textContent = data.completed_reps != null ? String(data.completed_reps) : '0';
      metricFrames.textContent = data.processed_frames != null ? String(data.processed_frames) : '0';

      const attemptsList = (data.attempts && data.attempts.events) ? data.attempts.events : [];
      metricAttempts.textContent = String(attemptsList.length);

      const uncountedCount = attemptsList.filter(e => e.status === 'uncounted').length;
      metricUncounted.textContent = String(uncountedCount);

      if (data.status === 'incomplete') {
        incompleteNoteText.textContent = data.read_status_note || 'Processing ended before declared video duration.';
        incompleteNoteBox.classList.remove('hidden');
      } else {
        incompleteNoteBox.classList.add('hidden');
      }

      const feedback = data.feedback_observations || {};
      const feedbackSummary = feedback.summary || {};
      const sessionContainer = document.getElementById('session-observations');
      sessionContainer.textContent = '';
      const sessionTexts = [...(feedbackSummary.session_observations || [])];
      (feedbackSummary.comparisons || []).forEach(group => {
        sessionTexts.push(...(group.observations || []), ...(group.limitations || []));
      });
      if (!sessionTexts.length) sessionTexts.push('No within-session comparison is available for this analysis.');
      sessionTexts.forEach(text => {
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        sessionContainer.appendChild(paragraph);
      });
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
        Object.keys(data.available_artifacts).forEach(name => {
          const item = data.available_artifacts[name];
          if (item && item.available) {
            const a = document.createElement('a');
            a.href = item.download_url;
            a.className = 'btn-secondary';
            a.textContent = 'Download ' + name + (item.size_bytes ? ' (' + (item.size_bytes / 1024).toFixed(0) + ' KB)' : '');
            a.setAttribute('download', name);
            downloadsContainer.appendChild(a);
          }
        });
      }
    }
    const restoredJob = new URLSearchParams(window.location.search).get('job');
    if (restoredJob && /^[0-9a-f-]{36}$/i.test(restoredJob)) {
      currentJobId = restoredJob;
      startJobTracking(restoredJob, 'Previously uploaded video');
    }
  </script>
</body>
</html>
"""


def create_app(job_manager: Optional[JobManager] = None) -> Any:
    """FastAPI application factory for 2D Gym Pose Analyser.

    Allows injecting a configured JobManager for isolated unit and integration testing.
    """
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

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the self-contained single-page web interface."""
        return HTMLResponse(content=INDEX_HTML, status_code=200)

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
        if artifact_name not in ALLOWED_ARTIFACTS:
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
    print(f"Compute device for analysis child process: {args.device}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
