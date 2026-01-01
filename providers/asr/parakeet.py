import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.restore_from(r"d:\models\tts\gguf\parakeet-tdt-0.6b-v3.nemo")


asr_model.transcribe([r"C:\d\project\home_assistant_AI\temp_media\kitten_tts_sample.wav"])
asr_model.transcribe([r"C:\d\project\home_assistant_AI\temp_media\4.wav"])

def nemo_transcribe(data):
    output = asr_model.transcribe([data])
    return output[0].text














