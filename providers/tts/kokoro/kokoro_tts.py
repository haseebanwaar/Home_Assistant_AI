import time

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import numpy as np
import io

from line_profiler_pycharm import profile

pipeline = KPipeline(lang_code='a')

def run_kokoro(text, voice='bf_lily', sr=24000):
    generator = pipeline(
        text, voice=voice,
        speed=1, split_pattern=r'\n+'
    )
    audio_buffer = io.BytesIO()

    audio_chunks = [audio for (_, _, audio) in generator]

    if len(audio_chunks) > 1:
        audio = np.concatenate(audio_chunks, axis=0)
    else:
        audio = audio_chunks[0]

    sf.write(audio_buffer, audio, sr, format='WAV')
    audio_buffer.seek(0)
    return audio_buffer.read()

