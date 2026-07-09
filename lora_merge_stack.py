import comfy
from .lora_merge import LoraMerger

class LoraMergerStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_lora": ("LoRA",),  # <-- ПЕРЕИМЕНОВАНО
                "mode": (["add", "concat", "svd"], {"default": "add"}),
                "rank": ("INT", {"default": 16, "min": 1, "max": 320, "step": 1}),
                "threshold": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float32"}),
                "output_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "lora_2": ("LoRA",),
                "lora_3": ("LoRA",),
                "lora_4": ("LoRA",),
                "lora_5": ("LoRA",),
                "lora_6": ("LoRA",),
                "lora_7": ("LoRA",),
                "lora_8": ("LoRA",),
                "lora_9": ("LoRA",),
                "lora_10": ("LoRA",),
            }
        }
    RETURN_TYPES = ("LoRA",)
    FUNCTION = "merge_stack"
    CATEGORY = "lora_merge"

    def merge_stack(self, master_lora, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0,
                    lora_2=None, lora_3=None, lora_4=None, lora_5=None,
                    lora_6=None, lora_7=None, lora_8=None, lora_9=None, lora_10=None):
        loras = [master_lora]
        for lora in (lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8, lora_9, lora_10):
            if lora is not None:
                loras.append(lora)

        if len(loras) == 1:
            print("ℹ️ Only one LoRA provided, returning as-is")
            return (master_lora,)

        merged = loras[0]
        merger = LoraMerger()
        for i, lora in enumerate(loras[1:], start=2):
            print(f"🔄 Merging step {i-1}: step_{i-1} + lora_{i}")
            merged = merger.merge_loras(merged, mode, rank, threshold, device, dtype, output_scale, lora)
            merged = merged[0]

        print(f"✅ Merged {len(loras)} LoRAs into one")
        return (merged,)

    @classmethod
    def IS_CHANGED(cls, master_lora, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0,
                   lora_2=None, lora_3=None, lora_4=None, lora_5=None,
                   lora_6=None, lora_7=None, lora_8=None, lora_9=None, lora_10=None):
        import hashlib
        ids = [id(master_lora), id(lora_2), id(lora_3), id(lora_4), id(lora_5),
               id(lora_6), id(lora_7), id(lora_8), id(lora_9), id(lora_10)]
        key = f"{ids}_{mode}_{rank}_{threshold}_{device}_{dtype}_{output_scale}"
        return hashlib.md5(key.encode()).hexdigest()