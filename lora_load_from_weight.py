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
        # Извлекаем данные из структуры LoRA
        lora_weight = lora.get("lora", {})
        strength_model = lora.get("strength_model", 1.0)
        strength_clip = lora.get("strength_clip", 1.0)

        # Проверяем, нужно ли применять LoRA
        if strength_model == 0 and strength_clip == 0:
            print("ℹ️ Both strengths are 0, skipping LoRA")
            return (model, clip)

        # Проверяем, что у нас есть данные LoRA
        if not lora_weight:
            print("⚠️ Warning: Empty LoRA data")
            return (model, clip)

        try:
            print(f"🔄 Applying LoRA with model={strength_model}, clip={strength_clip}")
            # Применяем LoRA к модели и CLIP
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora_weight, strength_model, strength_clip
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