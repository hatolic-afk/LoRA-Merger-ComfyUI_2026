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
                "rank": ("INT", {
                    "default": 16, 
                    "min": 1,
                    "max": 320,
                    "step": 1,
                }),
                "threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "device": (["cuda", "cpu"], ),
                "dtype": (["float32", "float16", "bfloat16"], ),
            },
            "optional": {
                "lora_2": ("LoRA",),
            }
        }
    RETURN_TYPES = ("LoRA", )
    FUNCTION = "merge_loras"
    CATEGORY = "lora_merge"

    def merge_loras(self, lora_1, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float16", lora_2=None):
        # ВАЖНО: параметр lora_2 должен быть последним!
        print(f"🔍 lora_1 type: {type(lora_1)}")
        print(f"🔍 lora_2 type: {type(lora_2)}")
        print(f"🔍 lora_1 keys: {list(lora_1.keys()) if lora_1 else 'None'}")
        print(f"🔍 lora_2 keys: {list(lora_2.keys()) if lora_2 else 'None'}")
        
        # Проверяем что обе LoRA переданы
        if lora_2 is None:
            print("⚠️ WARNING: lora_2 is None! Using only lora_1")
            return (lora_1,)
            
        # Проверяем что в lora_2 есть данные
        if not lora_2.get("lora", {}):
            print("⚠️ WARNING: lora_2 has no data! Using only lora_1")
            return (lora_1,)
            
        result = self.merge(lora_1, lora_2, mode, rank, threshold, device, dtype)
        
        # Проверяем результат
        if not result.get("lora", {}):
            print("⚠️ WARNING: Merged LoRA is empty! Using lora_1 as fallback")
            return (lora_1,)
            
        print(f"✅ Merged LoRA has {len(result['lora'])} keys")
        return (result,)
    
    @torch.no_grad()
    def merge(self, lora_1, lora_2, mode, rank, threshold, device, dtype):
        weight = {}
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        dtype = dtype_map.get(dtype, torch.float32)
        
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA not available, using CPU")
            device = "cpu"

        # Извлекаем данные
        lora_1_data = lora_1.get("lora", {})
        lora_1_model = lora_1.get("strength_model", 1.0)
        lora_1_clip = lora_1.get("strength_clip", 1.0)

        lora_2_data = lora_2.get("lora", {})
        lora_2_model = lora_2.get("strength_model", 1.0)
        lora_2_clip = lora_2.get("strength_clip", 1.0)

        print(f"📊 LoRA 1: {len(lora_1_data)} keys, model={lora_1_model}, clip={lora_1_clip}")
        print(f"📊 LoRA 2: {len(lora_2_data)} keys, model={lora_2_model}, clip={lora_2_clip}")

        # Проверяем данные
        if not lora_1_data and not lora_2_data:
            print("❌ ERROR: No LoRA data to merge!")
            return {"lora": {}, "strength_model": 1, "strength_clip": 1}

        # Получаем ключи
        keys_1 = [key[: key.rfind(".lora_down")] for key in lora_1_data.keys() if ".lora_down" in key]
        keys_2 = [key[: key.rfind(".lora_down")] for key in lora_2_data.keys() if ".lora_down" in key]
        
        # Если вторая пустая, используем только первую
        if not keys_2:
            print("ℹ️ Only one LoRA provided, using it directly")
            return {"lora": lora_1_data, "strength_model": lora_1_model, "strength_clip": lora_1_clip}
            
        keys = list(set(keys_1 + keys_2))
        
        if not keys:
            print("❌ ERROR: No valid LoRA keys found!")
            return {"lora": {}, "strength_model": 1, "strength_clip": 1}
            
        print(f"🔀 Merging {len(keys)} modules")
        print(f"  • Only in lora_1: {len(set(keys_1) - set(keys_2))}")
        print(f"  • Only in lora_2: {len(set(keys_2) - set(keys_1))}")
        print(f"  • Common: {len(set(keys_1) & set(keys_2))}")
        
        pber = comfy.utils.ProgressBar(len(keys))
        merged_count = 0

        for key in keys:
            try:
                up_key = key + ".lora_up.weight"
                down_key = key + ".lora_down.weight"
                alpha_key = key + ".alpha"
                
                # Определяем, какие данные использовать
                if key not in keys_1:
                    # Только в lora_2
                    print(f"  • {key}: only in lora_2")
                    up, down, alpha = self._get_up_down_alpha(key, lora_2_data, lora_2_model, lora_2_clip)
                    if mode == "svd":
                        up, down = self._svd_single(up, down, rank, threshold, device, dtype)
                        
                elif key not in keys_2:
                    # Только в lora_1
                    print(f"  • {key}: only in lora_1")
                    up, down, alpha = self._get_up_down_alpha(key, lora_1_data, lora_1_model, lora_1_clip)
                    if mode == "svd":
                        up, down = self._svd_single(up, down, rank, threshold, device, dtype)
                        
                else:
                    # В обеих LoRA - СМЕШИВАЕМ!
                    print(f"  • {key}: merging both")
                    up_1, down_1, alpha_1 = self._get_up_down_alpha(key, lora_1_data, lora_1_model, lora_1_clip)
                    up_2, down_2, alpha_2 = self._get_up_down_alpha(key, lora_2_data, lora_2_model, lora_2_clip)

                    # Приводим к одному типу
                    up_1 = up_1.to(dtype=dtype)
                    down_1 = down_1.to(dtype=dtype)
                    up_2 = up_2.to(dtype=dtype)
                    down_2 = down_2.to(dtype=dtype)

                    # Приводим размерности
                    if up_1.dim() != up_2.dim():
                        if up_1.dim() == 2 and up_2.dim() == 4:
                            up_2 = up_2.squeeze(2).squeeze(3)
                            down_2 = down_2.squeeze(2).squeeze(3)
                        elif up_1.dim() == 4 and up_2.dim() == 2:
                            up_1 = up_1.squeeze(2).squeeze(3)
                            down_1 = down_1.squeeze(2).squeeze(3)

                    if mode == "add":
                        # ПРОСТОЕ СЛОЖЕНИЕ - смешиваем девушку1 и девушку2
                        up = up_1 + up_2
                        down = down_1 + down_2
                        alpha = alpha_1
                        print(f"    • ADD: up_1 + up_2, down_1 + down_2")
                        
                    elif mode == "concat":
                        # КОНКАТЕНАЦИЯ
                        r_1 = up_1.shape[1]
                        r_2 = up_2.shape[1]
                        scale_1 = math.sqrt((r_1+r_2)/r_1) if r_1 > 0 else 1.0
                        scale_2 = math.sqrt((r_1+r_2)/r_2) if r_2 > 0 else 1.0
                        up = torch.cat([up_1*scale_1, up_2*scale_2], dim=1)
                        down = torch.cat([down_1*scale_1, down_2*scale_2], dim=0)
                        alpha = alpha_1 + alpha_2
                        print(f"    • CONCAT: r1={r_1}, r2={r_2}, new_rank={r_1+r_2}")
                        
                    elif mode == "svd":
                        # SVD слияние
                        up, down = self._svd_merge(up_1, down_1, up_2, down_2, rank, threshold, device)
                        alpha = torch.tensor(rank, dtype=torch.int64)
                        print(f"    • SVD: rank={rank}")

                # Сохраняем результаты
                weight[up_key] = up
                weight[down_key] = down
                weight[alpha_key] = alpha
                
                merged_count += 1
                pber.update(1)
                
            except Exception as e:
                print(f"❌ Error merging key {key}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Проверяем что есть результат
        if not weight:
            print("❌ ERROR: No weights were merged!")
            return {"lora": {}, "strength_model": 1, "strength_clip": 1}
            
        print(f"✅ Successfully merged {merged_count} modules with {len(weight)} total keys")
        return {"lora": weight, "strength_model": 1.0, "strength_clip": 1.0}

    def _get_up_down_alpha(self, key, lora_data, strength_model, strength_clip):
        """Извлекает up, down и alpha из данных LoRA"""
        up_key = key + ".lora_up.weight"
        down_key = key + ".lora_down.weight"
        alpha_key = key + ".alpha"

        if up_key not in lora_data or down_key not in lora_data:
            raise KeyError(f"Missing keys for {key}")

        is_te = "lora_te" in key
        scale = strength_clip if is_te else strength_model
        
        # Применяем масштабирование
        sqrt_scale = math.sqrt(abs(scale))
        sign_scale = 1 if scale >= 0 else -1

        up = lora_data[up_key] * sqrt_scale * sign_scale
        down = lora_data[down_key] * sqrt_scale
        
        # Получаем alpha
        if alpha_key in lora_data:
            alpha = lora_data[alpha_key]
        else:
            # Если alpha нет, используем ранг
            alpha = torch.tensor(up.shape[1], dtype=torch.int64)

        return up, down, alpha

    def _svd_single(self, up, down, rank, threshold, device, dtype):
        """SVD для одной LoRA"""
        org_device = up.device
        org_dtype = up.dtype
        
        up = up.to(device)
        down = down.to(device)
        r = up.shape[1]
        
        weight = up.view(-1, r) @ down.view(r, -1)
        weight = weight.to(dtype=torch.float32)
        
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        if threshold < 1.0:
            rank = self._index_sv_fro(S, threshold)
        rank = min(rank, len(S))
        
        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)
        Vh = Vh[:rank, :]
        
        # Clamp
        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val
        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)
        
        # Восстанавливаем размерность
        if down.dim() == 4:
            U = U.reshape(up.shape[0], rank, 1, 1)
            Vh = Vh.reshape(rank, down.shape[1], down.shape[2], down.shape[3])
        
        up = U.to(org_device, dtype=org_dtype) * math.sqrt(rank)
        down = Vh.to(org_device, dtype=org_dtype) * math.sqrt(rank)
        
        return up, down

    def _svd_merge(self, up_1, down_1, up_2, down_2, rank, threshold, device):
        """SVD слияние двух LoRA"""
        org_device = up_1.device
        org_dtype = up_1.dtype

        up_1 = up_1.to(device)
        down_1 = down_1.to(device)
        r_1 = up_1.shape[1]
        weight_1 = up_1.view(-1, r_1) @ down_1.view(r_1, -1)

        if up_2 is not None:
            up_2 = up_2.to(device)
            down_2 = down_2.to(device)
            r_2 = up_2.shape[1]
            weight_2 = up_2.view(-1, r_2) @ down_2.view(r_2, -1)
            weight = weight_1 / r_1 + weight_2 / r_2
        else:
            weight = weight_1 / r_1

        weight = weight.to(dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        if threshold < 1.0:
            rank = self._index_sv_fro(S, threshold)
        rank = min(rank, len(S))

        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)
        Vh = Vh[:rank, :]

        # Clamp
        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val
        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        if down_1.dim() == 4:
            U = U.reshape(up_1.shape[0], rank, 1, 1)
            Vh = Vh.reshape(rank, down_1.shape[1], down_1.shape[2], down_1.shape[3])

        up = U.to(org_device, dtype=org_dtype) * math.sqrt(rank)
        down = Vh.to(org_device, dtype=org_dtype) * math.sqrt(rank)

        return up, down

    def _index_sv_fro(self, S, target):
        """Находит индекс для сохранения target% энергии"""
        S_squared = S.pow(2)
        s_fro_sq = float(torch.sum(S_squared))
        if s_fro_sq == 0:
            return 1
        sum_S_squared = torch.cumsum(S_squared, dim=0)/s_fro_sq
        index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
        index = max(1, min(index, len(S)-1))
        return index

    @classmethod
    def IS_CHANGED(s, lora_1, mode="add", rank=16, threshold=1.0, device="cuda", dtype="float16", lora_2=None):
        import hashlib
        key_str = f"{id(lora_1)}_{id(lora_2)}_{mode}_{rank}_{threshold}_{device}_{dtype}"
        return hashlib.md5(key_str.encode()).hexdigest()