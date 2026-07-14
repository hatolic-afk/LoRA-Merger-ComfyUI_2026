"""
Слияние до 5 LoRA с правильной математикой.

Каждая лора уже содержит свой вес (из LoraLoaderWeightOnly).
Просто суммируем матрицы и делаем SVD/RSVD.
"""

import comfy
from .lora_merge import merge_multiple_loras


class LoraMergerMulti:
    """Слияние до 5 LoRA в одной ноде."""

    NUM_SLOTS = 5

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "target_rank": ("INT", {"default": 16, "min": 1, "max": 320, "step": 1}),
                "output_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "device":      (["cuda", "cpu"], {"default": "cuda"}),
                "dtype":       (["float32", "float16", "bfloat16"], {"default": "float32"}),
                "use_rsvd":    ("BOOLEAN", {"default": True}),
                "rsvd_oversampling": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "rsvd_n_iter": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            },
            "optional": {}
        }

        for i in range(1, cls.NUM_SLOTS + 1):
            inputs["optional"][f"lora_{i}"] = ("LoRA",)

        return inputs

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "merge_loras"
    CATEGORY = "lora_merge"

    def merge_loras(self, target_rank=16, output_power=1.0, device="cuda", dtype="float32",
                    use_rsvd=True, rsvd_oversampling=10, rsvd_n_iter=2, **kwargs):
        rsvd_oversampling = max(1, int(rsvd_oversampling))
        rsvd_n_iter = max(0, int(rsvd_n_iter))
        target_rank = max(1, int(target_rank))
        output_power = max(0.0, min(2.0, float(output_power)))

        loras = []
        for i in range(1, self.NUM_SLOTS + 1):
            lora = kwargs.get(f"lora_{i}", None)
            if lora is None:
                continue
            if not isinstance(lora, dict):
                continue
            lora_data = lora.get("lora", {})
            if not lora_data:
                continue
            loras.append(lora)

        if not loras:
            print("⚠️ No valid LoRAs provided")
            return ({"lora": {}},)

        if len(loras) == 1:
            print("ℹ️ Only one LoRA, returning as-is")
            return (loras[0],)

        print(f"🔀 Merging {len(loras)} LoRAs")

        result = merge_multiple_loras(
            loras, 
            [1.0] * len(loras),
            target_rank=target_rank, 
            device=device, 
            dtype=dtype,
            use_rsvd=use_rsvd, 
            rsvd_oversampling=rsvd_oversampling,
            rsvd_n_iter=rsvd_n_iter,
            output_power=output_power
        )
        
        return (result,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        import hashlib
        key_parts = []
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, dict):
                key_parts.append(f"{k}:{id(v)}")
            else:
                key_parts.append(f"{k}:{v}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()