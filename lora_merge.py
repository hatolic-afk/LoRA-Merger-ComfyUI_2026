import comfy
import math
import torch

CLAMP_QUANTILE = 0.99

class LoraMerger:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_1": ("LoRA",),
                "mode": (["add", "concat", "svd"], ),
                "rank": ("INT", {"default": 16, "min": 1, "max": 320, "step": 1}),
                "threshold": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
                "device": (["cuda", "cpu"], ),
                "dtype": (["float32", "float16", "bfloat16"], ),
                "output_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "lora_2": ("LoRA",),
            }
        }
    RETURN_TYPES = ("LoRA",)
    FUNCTION = "merge_loras"
    CATEGORY = "lora_merge"

    def merge_loras(self, lora_1, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0, lora_2=None):
        if lora_2 is None:
            print("⚠️ No lora_2 provided, returning lora_1 as-is")
            return (lora_1,)
        if not lora_2.get("lora", {}):
            print("⚠️ lora_2 has no data, returning lora_1")
            return (lora_1,)
        result = self.merge(lora_1, lora_2, mode, rank, threshold, device, dtype, output_scale)
        return (result,)

    @torch.no_grad()
    def merge(self, lora_1, lora_2, mode, rank, threshold, device, dtype, output_scale):
        weight = {}
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(dtype, torch.float32)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        l1_data = lora_1.get("lora", {})
        l2_data = lora_2.get("lora", {})

        # --- Универсальное определение суффиксов ---
        def detect_suffix_type(data):
            if any(".lora_up" in k or ".lora_down" in k for k in data):
                return "up/down"
            if any(".lora_A" in k or ".lora_B" in k for k in data):
                return "A/B"
            return None

        suffix1 = detect_suffix_type(l1_data)
        suffix2 = detect_suffix_type(l2_data)
        print(f"🔍 Detected suffix types: LoRA1={suffix1}, LoRA2={suffix2}")

        if suffix1 is None or suffix2 is None:
            print("❌ Could not detect suffix type – using only lora_1")
            return {"lora": l1_data, "strength_model": 1.0, "strength_clip": 1.0}

        # Функция для получения базового ключа
        def get_base_key(key, suffix_type):
            if suffix_type == "up/down":
                if ".lora_up" in key:
                    return key[:key.rfind(".lora_up")]
                if ".lora_down" in key:
                    return key[:key.rfind(".lora_down")]
            elif suffix_type == "A/B":
                if ".lora_A" in key:
                    return key[:key.rfind(".lora_A")]
                if ".lora_B" in key:
                    return key[:key.rfind(".lora_B")]
            return None

        keys_1 = list({get_base_key(k, suffix1) for k in l1_data.keys() if get_base_key(k, suffix1)})
        keys_2 = list({get_base_key(k, suffix2) for k in l2_data.keys() if get_base_key(k, suffix2)})
        all_keys = list(set(keys_1 + keys_2))

        print(f"🔀 Merging {len(all_keys)} modules")
        print(f"  keys_1: {len(keys_1)}, keys_2: {len(keys_2)}")

        if not keys_2:
            print("ℹ️ lora_2 has no base keys, using only lora_1")
            return {"lora": l1_data, "strength_model": 1.0, "strength_clip": 1.0}

        pbar = comfy.utils.ProgressBar(len(all_keys))

        for key in all_keys:
            try:
                # Формируем имена ключей
                if suffix1 == "up/down":
                    up_k1 = key + ".lora_up.weight"
                    down_k1 = key + ".lora_down.weight"
                    alpha_k1 = key + ".alpha"
                else:
                    up_k1 = key + ".lora_B.weight"
                    down_k1 = key + ".lora_A.weight"
                    alpha_k1 = key + ".alpha"

                if suffix2 == "up/down":
                    up_k2 = key + ".lora_up.weight"
                    down_k2 = key + ".lora_down.weight"
                    alpha_k2 = key + ".alpha"
                else:
                    up_k2 = key + ".lora_B.weight"
                    down_k2 = key + ".lora_A.weight"
                    alpha_k2 = key + ".alpha"

                has1 = up_k1 in l1_data and down_k1 in l1_data
                has2 = up_k2 in l2_data and down_k2 in l2_data

                if not has1 and not has2:
                    continue

                if not has1:
                    up, down, alpha = self._get_up_down_alpha_from_keys(key, l2_data, up_k2, down_k2, alpha_k2)
                    if mode == "svd":
                        up, down = self._svd_single(up, down, rank, threshold, device, dtype)
                elif not has2:
                    up, down, alpha = self._get_up_down_alpha_from_keys(key, l1_data, up_k1, down_k1, alpha_k1)
                    if mode == "svd":
                        up, down = self._svd_single(up, down, rank, threshold, device, dtype)
                else:
                    up1, down1, alpha1 = self._get_up_down_alpha_from_keys(key, l1_data, up_k1, down_k1, alpha_k1)
                    up2, down2, alpha2 = self._get_up_down_alpha_from_keys(key, l2_data, up_k2, down_k2, alpha_k2)

                    rank1 = up1.shape[1]
                    rank2 = up2.shape[1]

                    # Масштабирование второй с учётом alpha/rank
                    scale = math.sqrt((alpha2 / rank2) / (alpha1 / rank1))
                    up2_scaled = up2 * scale
                    down2_scaled = down2 * scale

                    up1 = up1.to(dtype=dtype)
                    down1 = down1.to(dtype=dtype)
                    up2_scaled = up2_scaled.to(dtype=dtype)
                    down2_scaled = down2_scaled.to(dtype=dtype)

                    # Приводим размерности
                    if up1.dim() != up2_scaled.dim():
                        if up1.dim() == 2 and up2_scaled.dim() == 4:
                            up2_scaled = up2_scaled.squeeze(2).squeeze(3)
                            down2_scaled = down2_scaled.squeeze(2).squeeze(3)
                        elif up1.dim() == 4 and up2_scaled.dim() == 2:
                            up1 = up1.squeeze(2).squeeze(3)
                            down1 = down1.squeeze(2).squeeze(3)

                    if mode == "add":
                        up = up1 + up2_scaled
                        down = down1 + down2_scaled
                        alpha = alpha1
                    elif mode == "concat":
                        r1 = up1.shape[1]
                        r2 = up2.shape[1]
                        s1 = math.sqrt((r1 + r2) / r1) if r1 else 1.0
                        s2 = math.sqrt((r1 + r2) / r2) if r2 else 1.0
                        up = torch.cat([up1 * s1, up2_scaled * s2], dim=1)
                        down = torch.cat([down1 * s1, down2_scaled * s2], dim=0)
                        alpha = alpha1 + alpha2
                    elif mode == "svd":
                        up, down = self._svd_merge(up1, down1, up2_scaled, down2_scaled, rank, threshold, device)
                        alpha = torch.tensor(rank, dtype=torch.int64)

                # Применяем output_scale ко всем up/down
                up = up * output_scale
                down = down * output_scale

                # Сохраняем с суффиксами первой LoRA
                if suffix1 == "up/down":
                    weight[key + ".lora_up.weight"] = up
                    weight[key + ".lora_down.weight"] = down
                else:
                    weight[key + ".lora_B.weight"] = up
                    weight[key + ".lora_A.weight"] = down
                weight[key + ".alpha"] = alpha

                pbar.update(1)
            except Exception as e:
                print(f"❌ Error on {key}: {e}")
                import traceback
                traceback.print_exc()

        if not weight:
            print("❌ No weights merged, returning lora_1")
            return {"lora": l1_data, "strength_model": 1.0, "strength_clip": 1.0}

        print(f"✅ Merged {len(weight)//3} modules with output_scale={output_scale}")
        return {"lora": weight, "strength_model": 1.0, "strength_clip": 1.0}

    def _get_up_down_alpha_from_keys(self, key, data, up_k, down_k, alpha_k):
        if up_k not in data or down_k not in data:
            raise KeyError(f"Missing keys for {key} (up={up_k}, down={down_k})")
        up = data[up_k]
        down = data[down_k]
        alpha = data.get(alpha_k, torch.tensor(up.shape[1], dtype=torch.int64))
        return up, down, alpha

    # Методы _svd_single, _svd_merge, _index_sv_fro остаются без изменений (скопируйте из предыдущей версии)
    # ...