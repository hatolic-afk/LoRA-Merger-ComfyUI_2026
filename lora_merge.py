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
                "master_lora": ("LoRA",),
                "mode": (["add", "concat", "svd", "weighted_avg", "weighted_sum", "interpolate", "magnitude", "difference"], ),
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

    def merge_loras(self, master_lora, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0, lora_2=None):
        if lora_2 is None:
            print("⚠️ No lora_2 provided, returning master_lora as-is")
            return (master_lora,)
        if not lora_2.get("lora", {}):
            print("⚠️ lora_2 has no data, returning master_lora")
            return (master_lora,)
        result = self.merge(master_lora, lora_2, mode, rank, threshold, device, dtype, output_scale)
        return (result,)

    @torch.no_grad()
    def merge(self, master_lora, lora_2, mode, rank, threshold, device, dtype, output_scale):
        weight = {}
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(dtype, torch.float32)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        l1_data = master_lora.get("lora", {})
        l2_data = lora_2.get("lora", {})

        def detect_suffix_type(data):
            if any(".lora_up" in k or ".lora_down" in k for k in data):
                return "up/down"
            if any(".lora_A" in k or ".lora_B" in k for k in data):
                return "A/B"
            return None

        suffix1 = detect_suffix_type(l1_data)
        suffix2 = detect_suffix_type(l2_data)
        print(f"🔍 Detected suffix types: master_lora={suffix1}, lora_2={suffix2}")

        if suffix1 is None or suffix2 is None:
            print("❌ Could not detect suffix type – using only master_lora")
            return {"lora": l1_data}

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

        print(f"🔀 Merging {len(all_keys)} modules using mode: {mode}")
        print(f"  master_lora keys: {len(keys_1)}, lora_2 keys: {len(keys_2)}")

        if not keys_2:
            print("ℹ️ lora_2 has no base keys, using only master_lora")
            return {"lora": l1_data}

        pbar = comfy.utils.ProgressBar(len(all_keys))

        for key in all_keys:
            try:
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

                    if up1.dim() != up2.dim():
                        if up1.dim() == 2 and up2.dim() == 4:
                            up2 = up2.squeeze(2).squeeze(3)
                            down2 = down2.squeeze(2).squeeze(3)
                        elif up1.dim() == 4 and up2.dim() == 2:
                            up1 = up1.squeeze(2).squeeze(3)
                            down1 = down1.squeeze(2).squeeze(3)

                    up1 = up1.to(dtype=dtype)
                    down1 = down1.to(dtype=dtype)
                    up2 = up2.to(dtype=dtype)
                    down2 = down2.to(dtype=dtype)

                    # ============= ОСНОВНАЯ ЛОГИКА =============
                    if mode == "add":
                        # Просто складываем (веса уже масштабированы)
                        up = up1 + up2
                        down = down1 + down2
                        alpha = alpha1
                        
                        # Легкая нормализация
                        up_norm = torch.norm(up)
                        if up_norm > 10.0:
                            up = up / (up_norm / 5.0)
                        down_norm = torch.norm(down)
                        if down_norm > 10.0:
                            down = down / (down_norm / 5.0)

                    elif mode == "weighted_avg":
                        # Простое усреднение 50/50
                        up = (up1 + up2) / 2
                        down = (down1 + down2) / 2
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "weighted_sum":
                        # Сумма с нормализацией
                        up = up1 + up2
                        down = down1 + down2
                        alpha = (alpha1 + alpha2) / 2
                        
                        up_norm = torch.norm(up)
                        down_norm = torch.norm(down)
                        if up_norm > 1.0:
                            up = up / up_norm
                        if down_norm > 1.0:
                            down = down / down_norm

                    elif mode == "interpolate":
                        # Интерполяция 50/50
                        t = 0.5
                        theta = t * torch.pi / 2
                        up = up1 * torch.cos(theta) + up2 * torch.sin(theta)
                        down = down1 * torch.cos(theta) + down2 * torch.sin(theta)
                        alpha = (alpha1 + alpha2) / 2
                        
                        up_norm = torch.norm(up)
                        if up_norm > 0:
                            up = up / up_norm * torch.norm(up1)
                        down_norm = torch.norm(down)
                        if down_norm > 0:
                            down = down / down_norm * torch.norm(down1)

                    elif mode == "magnitude":
                        # Берем максимальные по модулю
                        up_abs1 = torch.abs(up1)
                        up_abs2 = torch.abs(up2)
                        up = torch.where(up_abs1 > up_abs2, up1, up2)
                        
                        down_abs1 = torch.abs(down1)
                        down_abs2 = torch.abs(down2)
                        down = torch.where(down_abs1 > down_abs2, down1, down2)
                        
                        alpha = alpha1 if alpha1 > alpha2 else alpha2

                    elif mode == "difference":
                        # Добавляем только значимые различия
                        diff_up = up2 - up1
                        diff_down = down2 - down1
                        
                        threshold_diff = 0.1
                        mask_up = torch.abs(diff_up) > threshold_diff
                        mask_down = torch.abs(diff_down) > threshold_diff
                        
                        up = up1 + diff_up * mask_up.float()
                        down = down1 + diff_down * mask_down.float()
                        alpha = alpha1

                    elif mode == "concat":
                        # Конкатенация
                        up = torch.cat([up1, up2], dim=1)
                        down = torch.cat([down1, down2], dim=0)
                        alpha = alpha1 + alpha2

                    elif mode == "svd":
                        up, down = self._svd_merge(up1, down1, up2, down2, rank, threshold, device)
                        alpha = torch.tensor(rank, dtype=torch.int64)

                # Применяем output_scale
                up = up * output_scale
                down = down * output_scale
                
                # Безопасный clamp
                up = torch.clamp(up, -5.0, 5.0)
                down = torch.clamp(down, -5.0, 5.0)

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
            print("❌ No weights merged, returning master_lora")
            return {"lora": l1_data}

        print(f"✅ Merged {len(weight)//3} modules with output_scale={output_scale}")
        return {"lora": weight}

    def _get_up_down_alpha_from_keys(self, key, data, up_k, down_k, alpha_k):
        if up_k not in data or down_k not in data:
            raise KeyError(f"Missing keys for {key} (up={up_k}, down={down_k})")
        up = data[up_k]
        down = data[down_k]
        alpha = data.get(alpha_k, torch.tensor(up.shape[1], dtype=torch.int64))
        return up, down, alpha

    def _svd_single(self, up, down, rank, threshold, device, dtype):
        org_device = up.device
        org_dtype = up.dtype
        up = up.to(device)
        down = down.to(device)
        r = up.shape[1]
        weight = up.view(-1, r) @ down.view(r, -1)
        weight = weight.to(torch.float32)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        if threshold < 1.0:
            rank = self._index_sv_fro(S, threshold)
        rank = min(rank, len(S))
        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)
        Vh = Vh[:rank, :]
        U = torch.clamp(U, -1.0, 1.0)
        Vh = torch.clamp(Vh, -1.0, 1.0)
        if down.dim() == 4:
            U = U.reshape(up.shape[0], rank, 1, 1)
            Vh = Vh.reshape(rank, down.shape[1], down.shape[2], down.shape[3])
        up = U.to(org_device, dtype=org_dtype)
        down = Vh.to(org_device, dtype=org_dtype)
        return up, down

    def _svd_merge(self, up1, down1, up2, down2, rank, threshold, device):
        org_device = up1.device
        org_dtype = up1.dtype
        up1 = up1.to(device)
        down1 = down1.to(device)
        up2 = up2.to(device)
        down2 = down2.to(device)
        r1 = up1.shape[1]
        r2 = up2.shape[1]
        weight = (up1.view(-1, r1) @ down1.view(r1, -1)) / r1 + (up2.view(-1, r2) @ down2.view(r2, -1)) / r2
        weight = weight.to(torch.float32)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        if threshold < 1.0:
            rank = self._index_sv_fro(S, threshold)
        rank = min(rank, len(S))
        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)
        Vh = Vh[:rank, :]
        U = torch.clamp(U, -1.0, 1.0)
        Vh = torch.clamp(Vh, -1.0, 1.0)
        if down1.dim() == 4:
            U = U.reshape(up1.shape[0], rank, 1, 1)
            Vh = Vh.reshape(rank, down1.shape[1], down1.shape[2], down1.shape[3])
        up = U.to(org_device, dtype=org_dtype)
        down = Vh.to(org_device, dtype=org_dtype)
        return up, down

    def _index_sv_fro(self, S, target):
        S_squared = S.pow(2)
        total = float(torch.sum(S_squared))
        if total == 0:
            return 1
        cumsum = torch.cumsum(S_squared, dim=0) / total
        idx = int(torch.searchsorted(cumsum, target**2)) + 1
        idx = max(1, min(idx, len(S)-1))
        return idx

    @classmethod
    def IS_CHANGED(s, master_lora, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0, lora_2=None):
        import hashlib
        return hashlib.md5(f"{id(master_lora)}_{id(lora_2)}_{mode}_{rank}_{threshold}_{device}_{dtype}_{output_scale}".encode()).hexdigest()