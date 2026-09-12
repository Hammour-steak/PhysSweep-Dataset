"""Real decoded media must agree with release geometry and time, even with valid hashes."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.rendering.video_encoding import require_video_contract


class MediaContractTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe unavailable')
    def test_decoded_contract_rejects_size_rate_count_and_variable_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, size, rate, frames, filter_args, error in (
                ('valid', '64x48', 24, 5, [], None),
                ('size', '32x32', 24, 5, [], 'dimensions differ'),
                ('rate', '64x48', 12, 5, [], 'frame rate differs'),
                ('count', '64x48', 24, 4, [], 'frame count differs'),
                # Same 5 frames, same endpoints and average FPS; unequal middle intervals.
                ('vfr', '64x48', 24, 5, ['-vf', "settb=1/24000,setpts='if(eq(N,2),2500,N*1000)'"], 'timestamps differ'),
                ('offset', '64x48', 24, 5, ['-vf', 'setpts=PTS+1/TB'], 'timestamps differ'),
            ):
                with self.subTest(mode=mode):
                    video = root / f'{mode}.mp4'
                    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                                    f'color=c=blue:s={size}:r={rate}', '-frames:v', str(frames),
                                    *filter_args, '-vsync', '0', '-enc_time_base', '1:24000', '-c:v', 'libx264', '-bf', '0', '-threads', '1', str(video)], check=True)
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            require_video_contract(video, 5, resolution=[64, 48], fps=24)
                    else:
                        self.assertEqual(require_video_contract(video, 5, resolution=[64, 48], fps=24), 5)


if __name__ == '__main__':
    unittest.main()
