import soundfile as sf
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("/mnt/f/try/voxcpm")

tim=time.perf_counter()

wav = model.generate(
    text="relationship today are mix of memes, miscommunication and mystery, one day its good morning love and next silence,swiping left and right feels like a full time job and ghosting, sadly its just part of the game.love in the digital age exciting confusing and sometimes plain exhausting, Yeah!",
    # prompt_wav_path=None,      # optional: path to a prompt speech for voice cloning
    # prompt_wav_path=r"/mnt/c/d/project/home_assistant_AI/providers/outputs/4.wav",      # optional: path to a prompt speech for voice cloning
    prompt_wav_path=r"/mnt/f/try/kitten/audio.wav",      # optional: path to a prompt speech for voice cloning
    prompt_text="Google Colab: This is an indispensable tool for any AI student. Google Colab provides a free Jupyter Notebook environment that runs entirely in the cloud. Most importantly, it offers free access to GPU and TPU hardware accelerators. This allows you to train complex deep learning models without needing a powerful computer of your own.",          # optional: reference text
    # prompt_text="relationship today are mix of memes, miscommunication and mystery, one day its good morning love and next silence,swiping left and right feels like a full time job and ghosting, sadly its just part of the game.love in the digital age exciting confusing and sometimes plain exhausting, Yeah!",          # optional: reference text
    # prompt_text=None,          # optional: reference text
    cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
    normalize=True,           # enable external TN tool
    denoise=True,             # enable external Denoise tool
    retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
    retry_badcase_max_times=3,  # maximum retrying times
    retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
)
print(time.perf_counter()-tim)


sf.write("/mnt/f/try/output.wav", wav, 16000)
print("saved: output.wav")



