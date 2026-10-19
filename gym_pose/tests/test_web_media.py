from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from gym_analyser.web_media import create_preview


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'annotated.mp4'; self.source.write_bytes(b'original')
        self.dest = self.root/'preview.mp4'

    def test_missing_encoder_preserves_original(self):
        with patch('gym_analyser.web_media.shutil.which', return_value=None):
            self.assertFalse(create_preview(self.source,self.dest)['available'])
        self.assertEqual(self.source.read_bytes(),b'original')
        self.assertFalse(self.dest.exists())

    def test_success_publishes_separate_preview(self):
        def run(command,**kwargs):
            Path(command[-1]).write_bytes(b'encoded')
            self.assertIn('libx264',command); self.assertIn('-n',command)
            return subprocess.CompletedProcess(command,0)
        with patch('gym_analyser.web_media.shutil.which',return_value='ffmpeg'), patch('gym_analyser.web_media.subprocess.run',side_effect=run):
            self.assertTrue(create_preview(self.source,self.dest)['available'])
        self.assertEqual(self.dest.read_bytes(),b'encoded')
        self.assertEqual(self.source.read_bytes(),b'original')

    def test_timeout_leaves_analysis_intact(self):
        with patch('gym_analyser.web_media.shutil.which',return_value='ffmpeg'), patch('gym_analyser.web_media.subprocess.run',side_effect=subprocess.TimeoutExpired('ffmpeg',1)):
            self.assertFalse(create_preview(self.source,self.dest)['available'])
        self.assertFalse(self.dest.exists())

    def test_no_overwrite_or_outside_destination(self):
        self.dest.write_bytes(b'keep')
        with self.assertRaises(FileExistsError): create_preview(self.source,self.dest)
        with self.assertRaises(ValueError): create_preview(self.source,self.root/'outside'/'preview.mp4')
        self.assertEqual(self.dest.read_bytes(),b'keep')
