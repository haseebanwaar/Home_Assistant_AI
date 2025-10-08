from faster_whisper import WhisperModel
from line_profiler_pycharm import profile

modelUrdu = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")



@profile
def whisper_transcribe(data):
    segments, info = modelUrdu.transcribe(data, vad_filter=False,
                                          language='en',
                                          without_timestamps=True)
    transcription = ''
    for segment in segments:

        transcription += segment.text
    return transcription













