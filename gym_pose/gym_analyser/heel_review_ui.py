"""Experimental heel-review controls for the local results page."""
HTML = """
      <section class="card" id="heel-review-card" style="margin-top:16px" aria-labelledby="heel-review-title">
        <h3 id="heel-review-title">Experimental heel-lift review</h3>
        <p>Review flags only: accuracy is not established across different people or recordings. This does not grade safe or unsafe form.</p>
        <p>Choose 0.4 to 5 seconds of the video where you are standing still and both heels are flat and visible. Use the video above or below to check the times.</p>
        <p>Enter times in seconds, not angles. This interval selects a standing reference; the review then scans the video, not just that interval.</p>
        <form id="heel-review-form">
          <label>Reference start (seconds) <input id="heel-start" type="number" min="0" max="1200" step="0.1" required style="max-width:110px" aria-label="Flat-foot reference start in seconds"></label>
          <label>Reference end (seconds) <input id="heel-end" type="number" min="0" max="1200" step="0.1" required style="max-width:110px" aria-label="Flat-foot reference end in seconds"></label>
          <label style="display:block"><input id="heel-side-view" type="checkbox" required> I am filmed from the side.</label>
          <label style="display:block"><input id="heel-stationary" type="checkbox" required> The camera stays stationary.</label>
          <label style="display:block"><input id="heel-flat" type="checkbox" required> Both heels are flat throughout the selected reference.</label>
          <button type="submit" id="heel-submit" class="btn" style="margin-top:8px">Run experimental heel review</button>
        </form>
        <div id="heel-status" role="status" aria-live="polite">No heel review requested.</div>
        <div id="heel-events"></div>
        <a id="heel-download" style="color:var(--primary)" hidden>Download heel review JSON</a>
        <p>Inward-knee tracking and back rounding are not assessed by this heel review.</p>
      </section>
"""
JS = """
    let heelPollToken = 0;
    async function heelFetch(url, options) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);
      try {
        const response = await fetch(url, Object.assign({}, options || {}, {signal: controller.signal}));
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Heel review request failed.');
        return data;
      } finally { clearTimeout(timeout); }
    }
    function renderHeelReview(jobId, state) {
      const message = document.getElementById('heel-status');
      const events = document.getElementById('heel-events');
      const download = document.getElementById('heel-download');
      const submit = document.getElementById('heel-submit');
      if (!message || !events || !download || !submit) return;
      events.textContent = ''; download.hidden = true;
      submit.disabled = state.status === 'running';
      if (state.status === 'running') { message.textContent = 'Review running. Repetition results remain unchanged.'; return; }
      if (state.status === 'failed') { message.textContent = state.error || 'Heel review failed. Repetition results remain unchanged.'; return; }
      const result = state.result;
      if (!result) { message.textContent = 'No heel review requested.'; return; }
      const flags = Array.isArray(result.events) ? result.events : [];
      const measurable = Number(result.measurable_samples) || 0;
      const total = Number(result.total_samples) || 0;
      if (result.status === 'unable_to_assess') {
        message.textContent = 'Unable to assess: ' + String(result.reason || 'insufficient evidence').replace(/_/g, ' ') + '. Choose another clear, stationary flat-foot reference if appropriate.';
      } else {
        message.textContent = flags.length + ' possible heel-rise review flags. ' + measurable + '/' + total + ' sampled frames measurable. Accuracy unvalidated; no flag does not prove no heel lift.';
      }
      flags.forEach(flag => {
        const peak = Number(flag.peak_s), start = Number(flag.start_s), end = Number(flag.end_s);
        if (![peak,start,end].every(Number.isFinite) || start < 0 || end < start || peak < start || peak > end) return;
        const button = document.createElement('button'); button.type = 'button'; button.className = 'btn-seek';
        button.textContent = 'Review possible heel lift: ' + start.toFixed(2) + '-' + end.toFixed(2) + 's';
        button.addEventListener('click', () => seekVideoToTime(peak)); events.appendChild(button);
      });
      download.href = '/api/jobs/' + encodeURIComponent(jobId) + '/download/heel_review.json';
      download.setAttribute('download', 'heel_review.json'); download.hidden = false;
    }
    async function refreshHeelReview(jobId, poll) {
      const token = ++heelPollToken; const deadline = Date.now() + 210000;
      try {
        while (token === heelPollToken && currentJobId === jobId) {
          const state = await heelFetch('/api/jobs/' + encodeURIComponent(jobId) + '/heel-review');
          if (token !== heelPollToken || currentJobId !== jobId) return;
          renderHeelReview(jobId, state);
          if (state.status !== 'running' || Date.now() > deadline) return;
          await new Promise(resolve => setTimeout(resolve, 1500));
        }
      } catch (error) {
        const message = document.getElementById('heel-status');
        if (message && currentJobId === jobId) message.textContent = String(error.message || 'Heel review unavailable.');
        const button = document.getElementById('heel-submit'); if (button) button.disabled = false;
      }
    }
    const heelForm = document.getElementById('heel-review-form');
    if (heelForm && typeof heelForm.addEventListener === 'function') heelForm.addEventListener('submit', async event => {
      event.preventDefault(); if (!currentJobId) return;
      const jobId = currentJobId;
      const body = {
        reference_start_s: Number(document.getElementById('heel-start').value),
        reference_end_s: Number(document.getElementById('heel-end').value),
        side_view: document.getElementById('heel-side-view').checked,
        stationary_camera: document.getElementById('heel-stationary').checked,
        heels_flat: document.getElementById('heel-flat').checked
      };
      const button = document.getElementById('heel-submit'); const message = document.getElementById('heel-status');
      button.disabled = true; message.textContent = 'Starting experimental review...';
      try {
        await heelFetch('/api/jobs/' + encodeURIComponent(jobId) + '/heel-review', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
        await refreshHeelReview(jobId, true);
      } catch (error) { message.textContent = String(error.message || 'Heel review request failed.'); button.disabled = false; }
    });
"""


def add_heel_review_ui(html):
    html=html.replace('      <!-- Attempt Observations Table -->',HTML+'\n      <!-- Attempt Observations Table -->',1)
    html=html.replace("      'coaching_feedback.json',","      'coaching_feedback.json',\n      'heel_review.json',",1)
    html=html.replace('      renderCoachingFeedback(data.coaching_feedback);','      renderCoachingFeedback(data.coaching_feedback);\n      if (currentJobId) refreshHeelReview(currentJobId, false);',1)
    html=html.replace('    let restoredJob = null;',JS+'\n    let restoredJob = null;',1)
    return html
