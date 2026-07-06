from .lora_merge import LoraMerger
from .lora_load_from_weight import LoraLoaderFromWeight
from .lora_load_weight_only import LoraLoaderWeightOnly
from .lora_save import LoraSaveToFile

NODE_CLASS_MAPPINGS = {
    "LoraMerge": LoraMerger,
    "LoraLoaderFromWeight": LoraLoaderFromWeight,
    "LoraLoaderWeightOnly": LoraLoaderWeightOnly,
    "LoraSaveToFile": LoraSaveToFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraMerge": "Merge LoRA",
    "LoraLoaderFromWeight": "Load LoRA from Weight",
    "LoraLoaderWeightOnly": "Load LoRA Weight Only",
    "LoraSaveToFile": "Save LoRA to File",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']