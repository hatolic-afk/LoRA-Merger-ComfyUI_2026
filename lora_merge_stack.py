import comfy
from .lora_merge import LoraMerger

class LoraMergerStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_1": ("LoRA",),
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
            }
        }
    RETURN_TYPES = ("LoRA",)
    FUNCTION = "merge_stack"
    CATEGORY = "lora_merge"

    def merge_stack(self, lora_1, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0,
                    lora_2=None, lora_3=None, lora_4=None, lora_5=None):
        loras = [lora_1]
        for lora in (lora_2, lora_3, lora_4, lora_5):
            if lora is not None:
                loras.append(lora)

        if len(loras) == 1:
            print("ℹ️ Only one LoRA provided, returning as-is")
            return (lora_1,)

        merged = loras[0]
        merger = LoraMerger()
        for i, lora in enumerate(loras[1:], start=2):
            print(f"🔄 Merging step {i-1}: lora_{i-1} + lora_{i}")
            merged = merger.merge_loras(merged, mode, rank, threshold, device, dtype, output_scale, lora)
            merged = merged[0]

        print(f"✅ Merged {len(loras)} LoRAs into one")
        return (merged,)

    @classmethod
    def IS_CHANGED(cls, lora_1, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0,
                   lora_2=None, lora_3=None, lora_4=None, lora_5=None):
        import hashlib
        ids = [id(lora_1), id(lora_2), id(lora_3), id(lora_4), id(lora_5)]
        key = f"{ids}_{mode}_{rank}_{threshold}_{device}_{dtype}_{output_scale}"
        return hashlib.md5(key.encode()).hexdigest()