"""Clear Teng sample suggestions and navigation; frozen model output unchanged."""
from .heel_review_ui import JS as BASE_JS
HTML='''
      <section class="card" id="teng-review-card" style="margin-top:16px" aria-labelledby="teng-review-title">
        <h3 id="teng-review-title">Teng-trained back and heel review</h3>
        <p>Scan the video for frames resembling Teng's back or heel categories. You do not enter angles or a reference interval here. Suggestions are not confirmed faults.</p>
        <form id="teng-review-form">
          <label style="display:block"><input id="teng-side-view" type="checkbox" required> This is a side-view recording.</label>
          <label style="display:block"><input id="teng-stationary" type="checkbox" required> The camera stays stationary.</label>
          <button type="submit" id="teng-submit" class="btn" style="margin-top:8px">Run Teng-trained review</button>
        </form>
        <div id="teng-status" role="status" aria-live="polite">No Teng review requested.</div>
        <p>Choose a suggestion to bring the paused video into view. Nearby flagged samples are grouped for navigation only; group times are not verified fault boundaries.</p>
        <div id="teng-events"></div>
        <a id="teng-download" style="color:var(--primary)" hidden>Download Teng review JSON</a>
        <details style="margin-top:12px"><summary>How to interpret these suggestions</summary>
          <p>Standing, turning and shirt shape can trigger matches. Scores are not fault probabilities; no match does not prove good form. Outside counted repetitions can include partial movements, setup or standing, so those suggestions are kept separate rather than discarded.</p>
          <p>Native image test: heel F1 0.84, back F1 0.90 on 919 measurable images; 50 of 969 were unknown. Different-person and video accuracy are not established. Teng has no inward-knee labels.</p>
        </details>
      </section>
'''

GROUP_JS = '''
    let tengCountedWindows = [];
    function setTengCountedWindows(attempts) {
      const rows = Array.isArray(attempts) ? attempts : (attempts && Array.isArray(attempts.events) ? attempts.events : []);
      tengCountedWindows = rows.filter(a => a.status === 'counted' && typeof a.start_time_s === 'number' && Number.isFinite(a.start_time_s) && typeof a.end_time_s === 'number' && Number.isFinite(a.end_time_s) && a.start_time_s >= 0 && a.end_time_s >= a.start_time_s).map(a => [a.start_time_s, a.end_time_s]);
    }
    function tengSampleContext(time) {
      if (!tengCountedWindows.length) return 'unknown';
      return tengCountedWindows.some(w => time >= w[0] && time <= w[1]) ? 'counted' : 'outside';
    }
    function groupTengSamples(flags) {
      const groups = [];
      for (const kind of ['heel', 'back']) {
        const valid = (Array.isArray(flags) ? flags : []).filter(f => f && f.type === 'teng_' + kind + '_image_match' && typeof f.peak_s === 'number' && Number.isFinite(f.peak_s) && f.peak_s >= 0).map(f => ({time:f.peak_s, context:tengSampleContext(f.peak_s)})).sort((a,b) => a.time-b.time);
        let active = null;
        for (const sample of valid) {
          if (!active || sample.time-active.end > 0.45 || sample.context !== active.context) {
            active = {kind, start:sample.time, end:sample.time, times:[sample.time], context:sample.context}; groups.push(active);
          } else if (sample.time !== active.end) { active.end = sample.time; active.times.push(sample.time); }
        }
      }
      return groups;
    }
'''

RENDER_JS = '''
    function renderTengReview(jobId, state) {
      const message = document.getElementById('teng-status'), events = document.getElementById('teng-events'), download = document.getElementById('teng-download'), submit = document.getElementById('teng-submit');
      if (!message || !events || !download || !submit) return;
      events.textContent = ''; download.hidden = true; submit.disabled = state.status === 'running';
      if (state.status === 'running') { message.textContent = 'Review running across the video. Repetition results remain unchanged.'; return; }
      if (state.status === 'failed') { message.textContent = state.error || 'Review failed. Repetition results remain unchanged.'; return; }
      const result = state.result;
      if (!result) { message.textContent = 'No Teng review requested.'; return; }
      const flags = Array.isArray(result.events) ? result.events : [], groups = groupTengSamples(flags);
      const measurable = Number(result.measurable_samples) || 0, total = Number(result.total_samples) || 0;
      message.textContent = result.status === 'unable_to_assess'
        ? 'Unable to assess: ' + String(result.reason || 'insufficient evidence').replace(/_/g, ' ') + '.'
        : flags.length + ' flagged samples, grouped into ' + groups.length + ' review suggestions. Both measurements were available at ' + measurable + ' of ' + total + ' sampled times. These are not confirmed faults.';
      for (const kind of ['heel', 'back']) {
        const list = groups.filter(g => g.kind === kind), section = document.createElement('div'), heading = document.createElement('h4'), grid = document.createElement('div');
        heading.textContent = kind === 'heel' ? 'Heel suggestions' : 'Back suggestions'; section.appendChild(heading); grid.className = 'teng-suggestion-grid';
        if (!list.length) { const empty = document.createElement('p'); empty.textContent = 'No flagged samples for this category. This does not establish absence.'; section.appendChild(empty); }
        for (const group of list.slice(0, 60)) {
          const row = document.createElement('div'), button = document.createElement('button'), detail = document.createElement('p');
          row.className = 'teng-suggestion'; button.type = 'button'; button.className = 'btn-seek'; button.setAttribute('data-teng-time', String(group.start)); button.setAttribute('aria-pressed', 'false');
          const title = kind === 'heel' ? 'Heel' : 'Back';
          button.textContent = title + ': ' + group.start.toFixed(2) + 's' + (group.end > group.start ? ' - ' + group.end.toFixed(2) + 's' : '');
          detail.textContent = group.times.length + (group.times.length === 1 ? ' sampled frame' : ' sampled frames') + ' | ' + (group.context === 'counted' ? 'Within a counted repetition' : group.context === 'outside' ? 'Outside counted repetitions (may be partial movement or setup)' : 'Repetition context unavailable');
          button.addEventListener('click', () => {
            events.querySelectorAll('button[data-teng-time]').forEach(b => b.setAttribute('aria-pressed', 'false')); button.setAttribute('aria-pressed', 'true');
            seekVideoToTime(group.start);
            const context = document.getElementById('video-review-context');
            if (context) { context.hidden = false; context.textContent = 'Selected ' + title.toLowerCase() + ' suggestion at ' + group.start.toFixed(2) + 's. ' + detail.textContent + '. This is a review marker, not a confirmed fault.'; }
          });
          row.appendChild(button); row.appendChild(detail); grid.appendChild(row);
        }
        if (list.length > 60) { const note = document.createElement('p'); note.textContent = 'First 60 groups shown for this category; JSON retains every raw sample.'; section.appendChild(note); }
        section.appendChild(grid); events.appendChild(section);
      }
      download.href = '/api/jobs/' + encodeURIComponent(jobId) + '/download/teng_review.json'; download.setAttribute('download', 'teng_review.json'); download.hidden = false;
    }
'''


def add_teng_review_ui(html):
    code=BASE_JS.replace('Heel','Teng').replace('heel','teng')
    code=code.replace("        reference_start_s: Number(document.getElementById('teng-start').value),\n        reference_end_s: Number(document.getElementById('teng-end').value),\n",'')
    code=code.replace("stationary_camera: document.getElementById('teng-stationary').checked,\n        tengs_flat: document.getElementById('teng-flat').checked","stationary_camera: document.getElementById('teng-stationary').checked")
    start=code.index('    function renderTengReview(');end=code.index('    async function refreshTengReview(',start)
    code=code[:start]+GROUP_JS+RENDER_JS+code[end:]
    html=html.replace('      <!-- Attempt Observations Table -->',HTML+'\n      <!-- Attempt Observations Table -->',1)
    # Keep Teng prominent; retain older methods under an explicit optional disclosure.
    start=html.index('      <section class="card" id="heel-review-card"');teng=html.index('      <section class="card" id="teng-review-card"',start);end=html.index('</section>',teng)+len('</section>')
    html=html[:start]+html[teng:end]+'\n      <details class="card advanced-reviews" id="advanced-reference-reviews"><summary>Advanced reference-based reviews (optional)</summary><p>Alternative experimental methods. Enter times in seconds to choose a standing reference, not angles. Each method scans the video using that reference.</p>\n'+html[start:teng]+'\n      </details>\n'+html[end:]
    html=html.replace("      'back_review.json',","      'back_review.json',\n      'teng_review.json',",1)
    html=html.replace('      renderCoachingFeedback(data.coaching_feedback);','      renderCoachingFeedback(data.coaching_feedback);\n      setTengCountedWindows(data.attempts);\n      if (currentJobId) refreshTengReview(currentJobId, false);',1)
    return html.replace('    let restoredJob = null;',code+'\n    let restoredJob = null;',1)
