"""Meaningful light-theme contrast and progressive-disclosure contracts."""
import unittest
from gym_analyser.web import INDEX_HTML
from gym_analyser.web_theme import CSS

def luminance(hexcolor):
 rgb=[int(hexcolor[i:i+2],16)/255 for i in (1,3,5)]
 rgb=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
 return sum(a*b for a,b in zip(rgb,[.2126,.7152,.0722]))
def contrast(a,b):
 lo,hi=sorted([luminance(a),luminance(b)])
 return (hi+.05)/(lo+.05)
class TestSimpleTheme(unittest.TestCase):
 def test_readable_text_and_status_pairs(self):
  for text,bg in [('#252d28','#f5f6f3'),('#59665d','#ffffff'),('#31594a','#ffffff'),('#ffffff','#31594a'),('#70520c','#fbf0d5'),('#245534','#e5f0e5'),('#9a2c23','#fbe9e6'),('#4b594e','#edf0eb')]:
   with self.subTest(text=text,bg=bg):self.assertGreaterEqual(contrast(text,bg),4.5)
 def test_control_edges_and_focus(self):
  self.assertGreaterEqual(contrast('#7d8d7f','#fafbf9'),3);self.assertIn('outline: 3px solid var(--primary)',CSS);self.assertIn('prefers-reduced-motion',CSS)
 def test_upload_precedes_optional_guidance(self):
  self.assertLess(INDEX_HTML.index('id="upload-card"'),INDEX_HTML.index('class="card recording-guidance"'));self.assertIn('<summary>Recording tips &amp; limitations</summary>',INDEX_HTML);self.assertIn('</details>',INDEX_HTML)
 def test_no_neon_gradient_and_no_external_font_dependency(self):
  self.assertNotIn('linear-gradient(',INDEX_HTML);self.assertNotIn('radial-gradient(',INDEX_HTML);self.assertNotIn('fonts.googleapis',INDEX_HTML);self.assertIn('--primary: #31594a',INDEX_HTML)
if __name__=='__main__':unittest.main()
