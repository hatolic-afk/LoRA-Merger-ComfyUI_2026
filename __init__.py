from .lora_merge import LoraMerger
from .lora_merge_multi import LoraMergerMulti
from .lora_load_weight_only import LoraLoaderWeightOnly
from .lora_load_from_weight import LoraLoaderFromWeight
from .lora_save import LoraSaveToFile

NODE_CLASS_MAPPINGS = {
    "LoraMerge":            LoraMerger,
    "LoraMergerMulti":      LoraMergerMulti,
    "LoraLoaderWeightOnly": LoraLoaderWeightOnly,
    "LoraLoaderFromWeight": LoraLoaderFromWeight,
    "LoraSaveToFile":       LoraSaveToFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraMerge":            "Merge LoRA (2 inputs)",
    "LoraMergerMulti":      "Merge LoRAs (up to 5, RSVD)",
    "LoraLoaderWeightOnly": "Load LoRA (Weight Only)",
    "LoraLoaderFromWeight": "Apply LoRA to Model/CLIP",
    "LoraSaveToFile":       "Save LoRA to File",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']