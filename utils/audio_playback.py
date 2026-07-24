"""Best-effort server-side audio playback for proactive insights.

The screen-capture machine is the single POC user's own PC, so proactive
insights are spoken through its speakers. If sounddevice isn't available we
degrade gracefully (log + skip) instead of crashing.
"""
import io
import logging

import soundfile as sf

logger = logging.getLogger("home_assistant")

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - optional dependency
    sd = None
    logger.info("sounddevice unavailable (%s); server playback disabled.", exc)


def play_wav_bytes(wav_bytes):
    """Play WAV bytes on the server's default output device (blocking)."""
    if sd is None:
        logger.debug("Skipping playback — sounddevice not installed.")
        return
    try:
        data, samplerate = sf.read(io.BytesIO(wav_bytes))
        sd.play(data, samplerate)
        sd.wait()
    except Exception as exc:
        logger.warning("Audio playback failed: %s", exc)
