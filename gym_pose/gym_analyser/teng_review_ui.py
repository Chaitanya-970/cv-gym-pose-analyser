"""Teng-trained review controls; scores remain dataset-label matches."""
from .heel_review_ui import JS as BASE_JS
HTML='''
      <section class="card" id="teng-review-card" style="margin-top:16px" aria-labelledby="teng-review-title">
        <h3 id="teng-review-title">Teng-trained back and heel review</h3>
        <p>Find sampled frames resembling the dataset's Bad Back or Bad Heel images. These are review suggestions, not confirmed technique faults. No standing reference or posture judgment is needed.</p>
        <p>Native image test: heel F1 0.84, back F1 0.90 on 919 measurable images; 50 of 969 were unknown. Different-person and video accuracy are not established. Teng has no inward-knee labels.</p>
        <form id="teng-review-form">
          <label style="display:block"><input id="teng-side-view" type="checkbox" required> This is a side-view recording.</label>
          <label style="display:block"><input id="teng-stationary" type="checkbox" required> The camera stays stationary.</label>
          <button type="submit" id="teng-submit" class="btn" style="margin-top:8px">Run Teng-trained review</button>
        </form>
        <div id="teng-status" role="status" aria-live="polite">No Teng review requested.</div>
        <div id="teng-events"></div>
        <a id="teng-download" style="color:var(--primary)" hidden>Download Teng review JSON</a>
        <p>Shirt shape can hide the spine. Standing and turning may trigger matches. Scores are not fault probabilities; no match does not prove good form. Replay buttons mark sampled frames, not verified event boundaries.</p>
      </section>
'''


def add_teng_review_ui(html):
    code=BASE_JS.replace('Heel','Teng').replace('heel','teng')
    code=code.replace("        reference_start_s: Number(document.getElementById('teng-start').value),\n        reference_end_s: Number(document.getElementById('teng-end').value),\n",'')
    code=code.replace("stationary_camera: document.getElementById('teng-stationary').checked,\n        tengs_flat: document.getElementById('teng-flat').checked","stationary_camera: document.getElementById('teng-stationary').checked")
    code=code.replace('possible teng-rise review flags.','Teng image-label matches.').replace('sampled frames measurable. Accuracy unvalidated; no flag does not prove no teng lift.','sampled frames have both measurements. Video accuracy unvalidated; no match does not prove good form.')
    code=code.replace('Choose another clear, stationary flat-foot reference if appropriate.','Use a clear stationary side-view video with visible feet and torso.')
    code=code.replace('flags.forEach(flag => {','flags.slice(0, 60).forEach(flag => {')
    code=code.replace("button.textContent = 'Review possible teng lift: ' + start.toFixed(2) + '-' + end.toFixed(2) + 's';","button.textContent = (flag.type === 'teng_back_image_match' ? 'Back image match: ' : 'Heel image match: ') + peak.toFixed(2) + 's';")
    code=code.replace("      flags.slice(0, 60).forEach(flag => {","      if (flags.length > 60) { const note = document.createElement('p'); note.textContent = 'First 60 matches shown; download contains every sampled result.'; events.appendChild(note); }\n      flags.slice(0, 60).forEach(flag => {")
    html=html.replace('      <!-- Attempt Observations Table -->',HTML+'\n      <!-- Attempt Observations Table -->',1)
    html=html.replace("      'back_review.json',","      'back_review.json',\n      'teng_review.json',",1)
    html=html.replace('      renderCoachingFeedback(data.coaching_feedback);','      renderCoachingFeedback(data.coaching_feedback);\n      if (currentJobId) refreshTengReview(currentJobId, false);',1)
    return html.replace('    let restoredJob = null;',code+'\n    let restoredJob = null;',1)
