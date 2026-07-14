"""
Применяет уже отмасштабированную LoRA к MODEL и CLIP.
"""

import comfy


class LoraLoaderFromWeight:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip":  ("CLIP",),
                "lora":  ("LoRA",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip":  ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora_from_weight"
    CATEGORY = "lora_merge"

    def load_lora_from_weight(self, model, clip, lora, strength_model=1.0, strength_clip=1.0):
        lora_weight = lora.get("lora", {}) if isinstance(lora, dict) else {}

        if not lora_weight:
            print("⚠️ Empty LoRA data")
            return (model, clip)

        try:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora_weight, strength_model, strength_clip)
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"❌ Error applying LoRA: {e}")
            import traceback
            traceback.print_exc()
            return (model, clip)

    @classmethod
    def IS_CHANGED(s, model, clip, lora, strength_model, strength_clip):
        import time
        return time.time()