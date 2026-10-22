"""Front-view knee controls sharing the tested optional-review UI template."""
from .heel_review_ui import HTML as BASE_HTML,JS as BASE_JS


def add_knee_review_ui(html):
    card=BASE_HTML.replace('Experimental heel-lift review','Experimental inward-knee review')
    card=card.replace('where you are standing still and both heels are flat and visible','where your feet, knees and hips are steady and visible')
    card=card.replace('I am filmed from the side.','I am filmed from the front.')
    card=card.replace('Both heels are flat throughout the selected reference.','My feet, knees and hips are steady in the standing reference.')
    card=card.replace('heel','knee').replace('knee-side-view','knee-front-view').replace('knee-flat','knee-standing')
    card=card.replace('Inward-knee tracking and back rounding are not assessed by this knee review.','Heel lift and back rounding are not assessed by this knee review.')
    code=BASE_JS.replace('Heel','Knee').replace('heel','knee')
    code=code.replace('side_view:','front_view:').replace('knees_flat:','standing_still:')
    code=code.replace('knee-side-view','knee-front-view').replace('knee-flat','knee-standing')
    code=code.replace('possible knee-rise','possible inward-knee').replace('no knee lift','no inward-knee movement')
    code=code.replace('flat-foot reference','standing reference')
    code=code.replace("'Review possible knee lift: '","'Review possible inward-knee tracking (' + String(flag.side || 'leg') + '): '")
    html=html.replace('      <!-- Attempt Observations Table -->',card+'\n      <!-- Attempt Observations Table -->',1)
    html=html.replace("      'heel_review.json',","      'heel_review.json',\n      'knee_review.json',",1)
    html=html.replace('      renderCoachingFeedback(data.coaching_feedback);','      renderCoachingFeedback(data.coaching_feedback);\n      if (currentJobId) refreshKneeReview(currentJobId, false);',1)
    html=html.replace('    let restoredJob = null;',code+'\n    let restoredJob = null;',1)
    return html
