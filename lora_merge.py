import comfy
import math
import torch
import numpy as np

class LoraMerger:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "master_lora": ("LoRA",),
                "mode": (["add", "concat", "svd", "weighted_avg", "weighted_sum", "interpolate", "magnitude", "difference"], {"default": "add"}),
                "rank": ("INT", {"default": 16, "min": 1, "max": 320, "step": 1}),
                "threshold": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float32"}),
                "output_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "interp_method": (["slerp", "linear", "cubic", "cosine"], {"default": "slerp"}),
                "interp_t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lora_2": ("LoRA",),
            }
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "merge_loras"
    CATEGORY = "lora_merge"

    def merge_loras(self, master_lora, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0, interp_method="slerp", interp_t=0.5, lora_2=None):
        if lora_2 is None:
            print("⚠️ No lora_2 provided, returning master_lora as-is")
            return (master_lora,)
        
        if not lora_2.get("lora", {}):
            print("⚠️ lora_2 has no data, returning master_lora")
            return (master_lora,)

        if not master_lora.get("lora", {}):
            print("⚠️ master_lora has no data, returning lora_2")
            return (lora_2,)

        result = self.merge(master_lora, lora_2, mode, rank, threshold, device, dtype, output_scale, interp_method, interp_t)
        return (result,)

    def _check_and_fix_tensors(self, tensor):
        """Проверка и исправление nan/inf значений"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        return tensor

    def _normalize_tensor(self, tensor, eps=1e-10):
        """Нормализация тензора"""
        norm = torch.norm(tensor)
        if norm > eps:
            return tensor / norm
        return tensor

    def _slerp(self, v1, v2, t, eps=1e-10):
        """Spherical Linear Interpolation (Сферическая интерполяция) - лучший для нейросетей"""
        # Нормализация
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        
        if norm1 < eps or norm2 < eps:
            return v1 * (1 - t) + v2 * t
        
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        
        # Вычисляем угол между векторами
        dot = torch.clamp(torch.sum(v1_norm * v2_norm), -1.0, 1.0)
        theta = torch.acos(dot) * t
        
        # Slerp формула
        if torch.abs(dot) < 1.0 - eps:
            sin_theta = torch.sin(theta)
            sin_theta_abs = torch.abs(sin_theta)
            if sin_theta_abs > eps:
                v1_coeff = torch.sin(theta * (1 - t)) / sin_theta
                v2_coeff = torch.sin(theta * t) / sin_theta
            else:
                v1_coeff = 1 - t
                v2_coeff = t
        else:
            v1_coeff = 1 - t
            v2_coeff = t
        
        result = v1_norm * v1_coeff + v2_norm * v2_coeff
        # Восстанавливаем масштаб
        result_norm = torch.norm(result)
        if result_norm > eps:
            result = result / result_norm * (norm1 * (1 - t) + norm2 * t)
        
        return result

    def _linear_interp(self, v1, v2, t):
        """Линейная интерполяция"""
        return v1 * (1 - t) + v2 * t

    def _cubic_interp(self, v1, v2, t):
        """Кубическая интерполяция - более плавная чем линейная"""
        t2 = t * t
        t3 = t2 * t
        return v1 * (2*t3 - 3*t2 + 1) + v2 * (-2*t3 + 3*t2)

    def _cosine_interp(self, v1, v2, t):
        """Косинусоидальная интерполяция - плавный S-образный переход"""
        t_cos = (1 - torch.cos(t * torch.pi)) / 2
        return v1 * (1 - t_cos) + v2 * t_cos

    def _interpolate(self, up1, down1, up2, down2, t, method="slerp"):
        """Интерполяция с выбором метода"""
        if method == "slerp":
            up = self._slerp(up1, up2, t)
            down = self._slerp(down1, down2, t)
        elif method == "linear":
            up = self._linear_interp(up1, up2, t)
            down = self._linear_interp(down1, down2, t)
        elif method == "cubic":
            up = self._cubic_interp(up1, up2, t)
            down = self._cubic_interp(down1, down2, t)
        elif method == "cosine":
            up = self._cosine_interp(up1, up2, t)
            down = self._cosine_interp(down1, down2, t)
        else:
            up = self._slerp(up1, up2, t)
            down = self._slerp(down1, down2, t)
        
        return up, down

    @torch.no_grad()
    def merge(self, master_lora, lora_2, mode, rank, threshold, device, dtype, output_scale, interp_method, interp_t):
        """Основной метод слияния"""
        weight = {}
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        target_dtype = dtype_map.get(dtype, torch.float32)
        
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

        keys_1 = set()
        for k in l1_data.keys():
            base = get_base_key(k, suffix1)
            if base:
                keys_1.add(base)
                
        keys_2 = set()
        for k in l2_data.keys():
            base = get_base_key(k, suffix2)
            if base:
                keys_2.add(base)
                
        all_keys = list(keys_1 | keys_2)

        print(f"🔀 Merging {len(all_keys)} modules using mode: {mode}")
        print(f"  master_lora keys: {len(keys_1)}, lora_2 keys: {len(keys_2)}")
        print(f"  Interpolation method: {interp_method}, t={interp_t}")

        if not keys_2:
            print("ℹ️ lora_2 has no base keys, using only master_lora")
            return {"lora": l1_data}

        pbar = comfy.utils.ProgressBar(len(all_keys))

        for key in all_keys:
            try:
                # Формируем ключи для первой лоры
                if suffix1 == "up/down":
                    up_k1 = key + ".lora_up.weight"
                    down_k1 = key + ".lora_down.weight"
                    alpha_k1 = key + ".alpha"
                else:
                    up_k1 = key + ".lora_B.weight"
                    down_k1 = key + ".lora_A.weight"
                    alpha_k1 = key + ".alpha"

                # Формируем ключи для второй лоры
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
                    # Только вторая лора
                    up = l2_data[up_k2].clone()
                    down = l2_data[down_k2].clone()
                    alpha = l2_data.get(alpha_k2, torch.tensor(up.shape[1], dtype=torch.int64))
                    if mode == "svd":
                        up, down = self._svd_single(up, down, rank, threshold, device)
                elif not has2:
                    # Только первая лора
                    up = l1_data[up_k1].clone()
                    down = l1_data[down_k1].clone()
                    alpha = l1_data.get(alpha_k1, torch.tensor(up.shape[1], dtype=torch.int64))
                    if mode == "svd":
                        up, down = self._svd_single(up, down, rank, threshold, device)
                else:
                    # Обе лоры есть - смешиваем
                    up1 = l1_data[up_k1].clone()
                    down1 = l1_data[down_k1].clone()
                    up2 = l2_data[up_k2].clone()
                    down2 = l2_data[down_k2].clone()
                    alpha1 = l1_data.get(alpha_k1, torch.tensor(up1.shape[1], dtype=torch.int64))
                    alpha2 = l2_data.get(alpha_k2, torch.tensor(up2.shape[1], dtype=torch.int64))

                    # Сохраняем форму
                    orig_shape1 = up1.shape
                    orig_shape2 = up2.shape
                    
                    # Приводим к 2D
                    if up1.dim() == 4:
                        up1 = up1.view(up1.shape[0], -1)
                        down1 = down1.view(down1.shape[0], -1)
                    if up2.dim() == 4:
                        up2 = up2.view(up2.shape[0], -1)
                        down2 = down2.view(down2.shape[0], -1)
                    
                    up1 = up1.to(device, dtype=target_dtype)
                    down1 = down1.to(device, dtype=target_dtype)
                    up2 = up2.to(device, dtype=target_dtype)
                    down2 = down2.to(device, dtype=target_dtype)

                    up1 = self._check_and_fix_tensors(up1)
                    down1 = self._check_and_fix_tensors(down1)
                    up2 = self._check_and_fix_tensors(up2)
                    down2 = self._check_and_fix_tensors(down2)

                    # Выравниваем ранги если нужно
                    rank1 = up1.shape[1]
                    rank2 = up2.shape[1]
                    
                    if rank1 != rank2:
                        max_rank = max(rank1, rank2)
                        if rank1 < max_rank:
                            pad_size = max_rank - rank1
                            up1 = torch.nn.functional.pad(up1, (0, pad_size))
                            down1 = torch.nn.functional.pad(down1, (0, pad_size))
                        if rank2 < max_rank:
                            pad_size = max_rank - rank2
                            up2 = torch.nn.functional.pad(up2, (0, pad_size))
                            down2 = torch.nn.functional.pad(down2, (0, pad_size))

                    # СМЕШИВАНИЕ
                    if mode == "add":
                        up = up1 + up2
                        down = down1 + down2
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "weighted_avg":
                        up = (up1 + up2) / 2
                        down = (down1 + down2) / 2
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "weighted_sum":
                        up = up1 + up2
                        down = down1 + down2
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "interpolate":
                        # ИНТЕРПОЛЯЦИЯ С ВЫБОРОМ МЕТОДА
                        up, down = self._interpolate(up1, down1, up2, down2, interp_t, interp_method)
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "magnitude":
                        up_abs1 = torch.abs(up1)
                        up_abs2 = torch.abs(up2)
                        up = torch.where(up_abs1 >= up_abs2, up1, up2)
                        
                        down_abs1 = torch.abs(down1)
                        down_abs2 = torch.abs(down2)
                        down = torch.where(down_abs1 >= down_abs2, down1, down2)
                        alpha = alpha1 if alpha1 >= alpha2 else alpha2

                    elif mode == "difference":
                        up = up1 + (up2 - up1) * 0.5
                        down = down1 + (down2 - down1) * 0.5
                        alpha = alpha1

                    elif mode == "concat":
                        up = torch.cat([up1, up2], dim=1)
                        down = torch.cat([down1, down2], dim=0)
                        alpha = alpha1 + alpha2

                    elif mode == "svd":
                        weight_matrix = up1 @ down1 + up2 @ down2
                        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
                        
                        if threshold < 1.0:
                            rank = self._index_sv_fro(S, threshold)
                        rank = min(rank, len(S))
                        
                        U = U[:, :rank]
                        S = S[:rank]
                        U = U @ torch.diag(S)
                        Vh = Vh[:rank, :]
                        
                        if len(orig_shape1) == 4:
                            try:
                                U = U.reshape(up1.shape[0], rank, 1, 1)
                                Vh = Vh.reshape(rank, down1.shape[1], down1.shape[2], down1.shape[3])
                            except:
                                pass
                        
                        up = U
                        down = Vh
                        alpha = torch.tensor(rank, dtype=torch.int64)

                    else:
                        print(f"⚠️ Unknown mode: {mode}, using add")
                        up = up1 + up2
                        down = down1 + down2
                        alpha = (alpha1 + alpha2) / 2

                    # Восстанавливаем форму
                    if len(orig_shape1) == 4:
                        try:
                            up = up.view(orig_shape1)
                            down = down.view(orig_shape1)
                        except:
                            pass
                    elif len(orig_shape2) == 4:
                        try:
                            up = up.view(orig_shape2)
                            down = down.view(orig_shape2)
                        except:
                            pass

                # Проверяем на nan/inf
                up = self._check_and_fix_tensors(up)
                down = self._check_and_fix_tensors(down)

                # Применяем output_scale
                up = up * output_scale
                down = down * output_scale

                # Записываем результат
                if suffix1 == "up/down":
                    weight[key + ".lora_up.weight"] = up.to(device, dtype=target_dtype)
                    weight[key + ".lora_down.weight"] = down.to(device, dtype=target_dtype)
                else:
                    weight[key + ".lora_B.weight"] = up.to(device, dtype=target_dtype)
                    weight[key + ".lora_A.weight"] = down.to(device, dtype=target_dtype)
                
                weight[key + ".alpha"] = alpha.to(device)

            except Exception as e:
                print(f"❌ Error on {key}: {e}")
                import traceback
                traceback.print_exc()
            
            pbar.update(1)

        if not weight:
            print("❌ No weights merged, returning master_lora")
            return {"lora": l1_data}

        print(f"✅ Merged {len(weight)//3} modules with output_scale={output_scale}")
        return {"lora": weight}

    def _svd_single(self, up, down, rank, threshold, device):
        org_device = up.device
        org_dtype = up.dtype
        
        up = up.to(device, dtype=torch.float32)
        down = down.to(device, dtype=torch.float32)
        
        weight = up @ down
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        if threshold < 1.0:
            rank = self._index_sv_fro(S, threshold)
        rank = min(rank, len(S))
        
        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)
        Vh = Vh[:rank, :]
        
        if len(up.shape) == 4:
            try:
                U = U.reshape(up.shape[0], rank, 1, 1)
                Vh = Vh.reshape(rank, down.shape[1], down.shape[2], down.shape[3])
            except:
                pass
        
        return U.to(org_device, dtype=org_dtype), Vh.to(org_device, dtype=org_dtype)

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
    def IS_CHANGED(s, master_lora, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float32", output_scale=1.0, interp_method="slerp", interp_t=0.5, lora_2=None):
        import hashlib
        key = f"{id(master_lora)}_{id(lora_2)}_{mode}_{rank}_{threshold}_{device}_{dtype}_{output_scale}_{interp_method}_{interp_t}"
        return hashlib.md5(key.encode()).hexdigest()