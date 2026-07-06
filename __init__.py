from .lora_merge import LoraMerger
from .lora_load_from_weight import LoraLoaderFromWeight
from .lora_load_weight_only import LoraLoaderWeightOnly
from .lora_save import LoraSaveToFile
from .lora_merge_stack import LoraMergerStack   # <-- новый импорт

NODE_CLASS_MAPPINGS = {
    "LoraMerge": LoraMerger,
    "LoraLoaderFromWeight": LoraLoaderFromWeight,
    "LoraLoaderWeightOnly": LoraLoaderWeightOnly,
    "LoraSaveToFile": LoraSaveToFile,
    "LoraMergerStack": LoraMergerStack,          # <-- новая нода
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraMerge": "Merge LoRA",
    "LoraLoaderFromWeight": "Load LoRA from Weight",
    "LoraLoaderWeightOnly": "Load LoRA Weight Only",
    "LoraSaveToFile": "Save LoRA to File",
    "LoraMergerStack": "Merge LoRA Stack (up to 5)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']