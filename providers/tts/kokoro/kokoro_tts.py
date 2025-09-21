from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import io

from line_profiler_pycharm import profile

pipeline = KPipeline(lang_code='a')

@profile
def run_kokoro(text, voice='bf_lily', sr=24000):

    generator = pipeline(
        text, voice=voice,  # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    audio_buffer = io.BytesIO()

    for i, (_, _, audio) in enumerate(generator):
        sf.write(audio_buffer, audio, sr, format='WAV')

        audio_buffer.seek(0)  # Rewind the buffer to the beginning
        assert i==0
    return audio_buffer.read()

    # sf.write(f'{i}.wav', audio, sr)  # save each audio file


#
# import time
# tim=time.perf_counter()
#
# generator = pipeline(
#     text, voice=voice,  # <= change voice here
#     speed=1, split_pattern=r'\n+'
# )
# audio_buffer = io.BytesIO()
#
# for i, (_, _, audio) in enumerate(generator):
#     sf.write('./output.wav', audio, sr)
#     print(i)
# print(time.perf_counter()-tim)
