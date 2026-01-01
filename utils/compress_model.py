#------------------------------
import requests
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration
from transformers import AutoProcessor,  InternVLProcessor,InternVLVideoProcessor
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from transformers import InternVLProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Load model.
model_id ="/mnt/c/keyes"  # or "HuggingFaceTB/SmolVLM-Instruct"
# model_id ="OpenGVLab/InternVL3_5-1B-HF"  # or "HuggingFaceTB/SmolVLM-Instruct"
# model_id ="f:/try/intern1"  # or "HuggingFaceTB/SmolVLM-Instruct"
model = AutoModel.from_pretrained(model_id,trust_remote_code=True,device_map="auto",torch_dtype=torch.bfloat16 )
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# v_processor = InternVLVideoProcessor.from_pretrained(model_id, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)




# Oneshot arguments
# DATASET_ID = "lmms-lab/flickr30k"
DATASET_ID = "Vishva007/Flickr-Dataset-1k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 4096  # Seems to be required here


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
        # "language_model.lm_head",
        # "language_model.model.embed_tokens",
        # "language_model.model.norm",
        # "re:.*input_layernorm",
        # "re:.*post_attention_layernorm",
        "re:vision_model.*",
        "re:visual.*",
        "re:vision_tower.*",  # skip ALL vision encoder layers
        "re:.*multi_modal_projector.*",
        "re:mlp1.*",
        "re:mlp_AR.*",
        "language_model.lm_head",
        "language_model.model.embed_tokens",],
    ),
]

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split='train')
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


# Apply chat template
def preprocess(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What does this image show?"},
                {"type": "image"},
            ],
        },
        {
            "role": "assistant",
            "content": " ".join(example["caption"]),
        },
    ]
    return {
        "text": processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
        ),
        "images": example["image"],
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return processor(
        text=sample["text"],
        images=sample["images"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )




# avoid errors with writer_batch_size
ds = ds.map(tokenize, writer_batch_size=1, remove_columns=ds.column_names)

# Perform oneshot

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    sequential_targets=["KeyeVL1_5DecoderLayer"],
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe the animal in this image\n"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1000)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)


