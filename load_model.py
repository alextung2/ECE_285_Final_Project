
import zipfile
import psutil
import torch
import transformers.modeling_utils as modeling_utils

from transformers import BitsAndBytesConfig
from medusa.model.medusa_model import MedusaModel

# Use memory mapping for PyTorch checkpoint shards.
original_load_state_dict = modeling_utils.load_state_dict

def load_state_dict_mmap(checkpoint_file):
    if str(checkpoint_file).endswith(".bin"):
        if not zipfile.is_zipfile(checkpoint_file):
            raise RuntimeError(
                "This checkpoint format does not support memory mapping."
            )

        return torch.load(
            checkpoint_file,
            map_location="cpu",
            mmap=True,
        )

    return original_load_state_dict(checkpoint_file)

modeling_utils.load_state_dict = load_state_dict_mmap

ram = psutil.virtual_memory()
print(f"Total system RAM: {ram.total / 1024**3:.2f} GiB")
print(f"Available system RAM: {ram.available / 1024**3:.2f} GiB")

model_id = "FasterDecoding/medusa-vicuna-7b-v1.3"

# Quantize the backbone while keeping prediction heads in FP16.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["lm_head", "medusa_head"],
)

print("Loading model with memory-mapped checkpoints...", flush=True)

try:
    model = MedusaModel.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        device_map={"": 0},
    )
finally:
    modeling_utils.load_state_dict = original_load_state_dict

model.eval()

print("\nModel loaded successfully")
print("Model type:", type(model).__name__)
print("Number of Medusa heads:", len(model.medusa_head))

allocated = torch.cuda.memory_allocated() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3

print(f"Current GPU tensor memory: {allocated:.2f} GiB")
print(f"Peak GPU tensor memory: {peak:.2f} GiB")
print("Model loading check completed")
