import comfy

class LoraLoaderFromWeight:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora": ("LoRA", ),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora_from_weight"
    CATEGORY = "lora_merge"

    def load_lora_from_weight(self, model, clip, lora):
        lora_weight = lora.get("lora", {})
        
        if not lora_weight:
            print("⚠️ Warning: Empty LoRA data")
            return (model, clip)

        try:
            print(f"🔄 Applying LoRA (weights already scaled)")
            # Веса уже масштабированы в LoraLoaderWeightOnly
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora_weight, 1.0, 1.0
            )
            print("✅ LoRA applied successfully")
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"❌ Error loading LoRA: {e}")
            import traceback
            traceback.print_exc()
            return (model, clip)

    @classmethod
    def IS_CHANGED(s, model, clip, lora):
        import time
        return time.time()