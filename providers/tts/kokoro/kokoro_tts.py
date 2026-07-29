from kokoro import KPipeline
import soundfile as sf
import numpy as np
import io
from threading import Lock

from .voice_settings import KokoroVoiceSettings

_voice_settings = KokoroVoiceSettings()
_pipeline_lock = Lock()
_initial_language = _voice_settings.voice[0]
pipeline = KPipeline(
    lang_code=_initial_language,
    repo_id='hexgrad/Kokoro-82M',
)
_pipelines = {_initial_language: pipeline}


def get_kokoro_voice_settings():
    return _voice_settings.payload()


def set_kokoro_voice(voice):
    return _voice_settings.set_voice(voice)


def _pipeline_for_voice(voice):
    language = voice[0]
    with _pipeline_lock:
        selected = _pipelines.get(language)
        if selected is None:
            selected = KPipeline(
                lang_code=language,
                repo_id='hexgrad/Kokoro-82M',
                model=pipeline.model,
            )
            _pipelines[language] = selected
        return selected


def run_kokoro(text, voice=None, sr=24000):
    selected_voice = voice or _voice_settings.voice
    selected_pipeline = _pipeline_for_voice(selected_voice)
    generator = selected_pipeline(
        text, voice=selected_voice,
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

