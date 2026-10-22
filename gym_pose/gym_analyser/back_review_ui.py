"""Experimental spine review controls using the shared optional-review template."""
from .heel_review_ui import HTML as BASE_HTML, JS as BASE_JS


def add_back_review_ui(html):
    card=BASE_HTML.replace('Experimental heel-lift review','Experimental back-rounding review')
    card=card.replace('where you are standing still and both heels are flat and visible','where you are standing still in a clear side view')
    card=card.replace('Both heels are flat throughout the selected reference.','I am standing still throughout the selected reference. This does not certify a neutral spine.')
    card=card.replace('Inward-knee tracking and back rounding are not assessed by this heel review.','This compares model-estimated lumbar curvature with your standing reference. It cannot confirm a technique fault. Heel lift and inward-knee tracking are assessed separately. The optional spine model is for non-commercial research.')
    card=card.replace('heel','back').replace('back-flat','back-standing').replace('Flat-foot','Standing')
    card=card.replace('          <button type="submit"','          <label style="display:block">Facing direction <select id="back-facing" required><option value="">Choose direction</option><option value="left">Left on screen</option><option value="right">Right on screen</option></select></label>\n          <label style="display:block"><input id="back-no-turning" type="checkbox" required> I keep facing the same direction throughout the video.</label>\n          <button type="submit"',1)
    code=BASE_JS.replace('Heel','Back').replace('heel','back')
    code=code.replace('backs_flat:','standing_still:').replace('back-flat','back-standing')
    code=code.replace("standing_still: document.getElementById('back-standing').checked", "standing_still: document.getElementById('back-standing').checked,\n        facing: document.getElementById('back-facing').value,\n        same_facing_direction: document.getElementById('back-no-turning').checked")
    code=code.replace('possible back-rise review flags','possible lumbar-curvature review flags').replace('no back lift','no back rounding')
    code=code.replace('flat-foot reference','standing reference').replace('Review possible back lift:','Review possible back rounding:')
    html=html.replace('      <!-- Attempt Observations Table -->',card+'\n      <!-- Attempt Observations Table -->',1)
    html=html.replace("      'knee_review.json',","      'knee_review.json',\n      'back_review.json',",1)
    html=html.replace('      renderCoachingFeedback(data.coaching_feedback);','      renderCoachingFeedback(data.coaching_feedback);\n      if (currentJobId) refreshBackReview(currentJobId, false);',1)
    return html.replace('    let restoredJob = null;',code+'\n    let restoredJob = null;',1)
