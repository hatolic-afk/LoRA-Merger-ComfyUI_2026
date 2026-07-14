"""
Загрузка одной LoRA с весом и LBW.
"""

import math
import os
import re
import folder_paths
import comfy.utils


def _extract_numbers(s):
    return [int(num) for num in re.findall(r'\d+', s)]


def _detect_block_keys(lora, n_blocks):
    block_keys = {}
    up_keys = [
        k for k in lora.keys()
        if ("lora_up" in k or "lora_B" in k) and "lora_te" not in k
    ]

    for key in up_keys:
        ids = _extract_numbers(key)

        if "input_blocks" in key and ids:
            block_id = ids[0]
        elif "middle_block" in key or "mid_block" in key:
            block_id = n_blocks // 2
        elif "output_blocks" in key and ids:
            block_id = ids[0] + (n_blocks // 2) + 1
        elif "down_blocks" in key and len(ids) >= 2:
            block_id = ids[0] * 3 + ids[1] + 1
            if any(t in key for t in ("downsampler", "down_sampler", ".op.")):
                block_id += 2
        elif "up_blocks" in key and len(ids) >= 2:
            block_id = ids[0] * 3 + ids[1] + (n_blocks // 2) + 1
            if any(t in key for t in ("upsampler", "up_sampler", ".op.")):
                block_id += 2
        else:
            block_id = 0

        if 0 <= block_id < n_blocks:
            block_keys.setdefault(block_id, []).append(key)

    return block_keys


def _parse_lbw(text):
    if not text:
        return []
    cleaned = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if not cleaned:
        return []
    out = []
    for tok in cleaned.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


class LoraLoaderWeightOnly:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "weight":    ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lbw":       ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "load_lora_weight_only"
    CATEGORY = "lora_merge"

    def load_lora_weight_only(self, lora_name, weight, lbw):
        lora_path = folder_paths.get_full_path("loras", lora_name)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path and self.loaded_lora[2] == (weight, lbw):
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                print(f"📂 Loaded {lora_name}: {len(lora)} keys")
            except Exception as e:
                print(f"❌ Error loading LoRA {lora_name}: {e}")
                return ({"lora": {}},)

            # LBW
            weight_list = _parse_lbw(lbw)
            if weight_list:
                is_xl_style = any("down_blocks" in k for k in lora.keys())
                n_blocks = 20 if is_xl_style else 26
                block_keys = _detect_block_keys(lora, n_blocks)

                for block_id, keys in block_keys.items():
                    if block_id >= len(weight_list):
                        continue
                    w = weight_list[block_id]
                    if abs(w) < 1e-10:
                        for key in keys:
                            lora.pop(key, None)
                            down_key = key.replace("lora_up", "lora_down").replace("lora_B", "lora_A")
                            lora.pop(down_key, None)
                            alpha_key = key.replace("lora_up.weight", "alpha").replace("lora_B.weight", "alpha")
                            lora.pop(alpha_key, None)
                    else:
                        abs_s = math.sqrt(abs(w))
                        sign = 1.0 if w >= 0 else -1.0
                        up_scale = abs_s * sign
                        down_scale = abs_s
                        for key in keys:
                            if key in lora:
                                lora[key] = lora[key] * up_scale
                            down_key = key.replace("lora_up", "lora_down").replace("lora_B", "lora_A")
                            if down_key in lora:
                                lora[down_key] = lora[down_key] * down_scale

            # Глобальный weight
            if abs(weight - 1.0) > 1e-10:
                abs_s = math.sqrt(abs(weight))
                sign = 1.0 if weight >= 0 else -1.0
                up_scale = abs_s * sign
                down_scale = abs_s

                suffix = None
                for k in lora:
                    if "lora_up" in k or "lora_down" in k:
                        suffix = "up/down"
                        break
                    if "lora_A" in k or "lora_B" in k:
                        suffix = "A/B"
                        break

                if suffix:
                    for k in list(lora.keys()):
                        if k.endswith(".alpha"):
                            continue
                        if suffix == "up/down" and "lora_up" in k:
                            lora[k] = lora[k] * up_scale
                        elif suffix == "up/down" and "lora_down" in k:
                            lora[k] = lora[k] * down_scale
                        elif suffix == "A/B" and "lora_B" in k:
                            lora[k] = lora[k] * up_scale
                        elif suffix == "A/B" and "lora_A" in k:
                            lora[k] = lora[k] * down_scale

            self.loaded_lora = (lora_path, lora, (weight, lbw))

        return ({"lora": lora},)

    @classmethod
    def IS_CHANGED(s, lora_name, weight, lbw):
        import hashlib
        key = f"{lora_name}_{weight}_{lbw}"
        return hashlib.md5(key.encode()).hexdigest()