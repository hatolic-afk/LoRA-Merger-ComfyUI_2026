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

        # Проверяем, что master_lora имеет данные
        if not master_lora.get("lora", {}):
            print("⚠️ master_lora has no data, returning lora_2")
            return (lora_2,)

        result = self.merge(master_lora, lora_2, mode, rank, threshold, device, dtype, output_scale)
        return (result,)

    def _align_tensors(self, up1, down1, up2, down2):
        """Выравнивает тензоры для смешивания"""
        # Если размерности совпадают - возвращаем как есть
        if up1.dim() == up2.dim() and up1.shape == up2.shape:
            return up1, down1, up2, down2
            
        # Обработка 2D vs 4D
        if up1.dim() == 2 and up2.dim() == 4:
            # Конвертируем 4D в 2D (сжимаем пространственные размеры)
            up2_flat = up2.view(up2.shape[0], -1)
            down2_flat = down2.view(down2.shape[0], -1)
            # После смешивания восстановим 4D
            return up1, down1, up2_flat, down2_flat
        elif up1.dim() == 4 and up2.dim() == 2:
            up1_flat = up1.view(up1.shape[0], -1)
            down1_flat = down1.view(down1.shape[0], -1)
            return up1_flat, down1_flat, up2, down2
            
        # Разные ранги - выравниваем через проекцию
        if up1.shape[1] != up2.shape[1]:
            # Проекция на больший ранг
            if up1.shape[1] < up2.shape[1]:
                # Дополняем нулями до большего ранга
                pad_size = up2.shape[1] - up1.shape[1]
                up1_pad = torch.nn.functional.pad(up1, (0, pad_size))
                down1_pad = torch.nn.functional.pad(down1, (0, pad_size))
                return up1_pad, down1_pad, up2, down2
            else:
                # Обрезаем до меньшего ранга
                up2_cut = up2[:, :up1.shape[1]]
                down2_cut = down2[:, :up1.shape[1]]
                return up1, down1, up2_cut, down2_cut
                
        return up1, down1, up2, down2

    def _restore_shape(self, tensor, original_shape):
        """Восстанавливает форму тензора"""
        if tensor.dim() == 2 and len(original_shape) == 4:
            return tensor.view(original_shape)
        return tensor

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

        # Получаем все уникальные ключи
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

        if not keys_2:
            print("ℹ️ lora_2 has no base keys, using only master_lora")
            return {"lora": l1_data}

        pbar = comfy.utils.ProgressBar(len(all_keys))

        for key in all_keys:
            try:
                # Формируем ключи для первого LoRA
                if suffix1 == "up/down":
                    up_k1 = key + ".lora_up.weight"
                    down_k1 = key + ".lora_down.weight"
                    alpha_k1 = key + ".alpha"
                else:
                    up_k1 = key + ".lora_B.weight"
                    down_k1 = key + ".lora_A.weight"
                    alpha_k1 = key + ".alpha"

                # Формируем ключи для второго LoRA
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
                    # Используем только второй
                    up = l2_data[up_k2].clone()
                    down = l2_data[down_k2].clone()
                    alpha = l2_data.get(alpha_k2, torch.tensor(up.shape[1], dtype=torch.int64))
                    if mode == "svd":
                        up, down = self._svd_single(up, down, rank, threshold, device, dtype)
                elif not has2:
                    # Используем только первый
                    up = l1_data[up_k1].clone()
                    down = l1_data[down_k1].clone()
                    alpha = l1_data.get(alpha_k1, torch.tensor(up.shape[1], dtype=torch.int64))
                    if mode == "svd":
                        up, down = self._svd_single(up, down, rank, threshold, device, dtype)
                else:
                    # Оба присутствуют - смешиваем
                    up1 = l1_data[up_k1].clone()
                    down1 = l1_data[down_k1].clone()
                    up2 = l2_data[up_k2].clone()
                    down2 = l2_data[down_k2].clone()
                    alpha1 = l1_data.get(alpha_k1, torch.tensor(up1.shape[1], dtype=torch.int64))
                    alpha2 = l2_data.get(alpha_k2, torch.tensor(up2.shape[1], dtype=torch.int64))

                    # Сохраняем оригинальные формы для восстановления
                    orig_shape1 = up1.shape if up1.dim() == 4 else None
                    orig_shape2 = up2.shape if up2.dim() == 4 else None

                    # Приводим к единому формату для смешивания
                    up1_flat, down1_flat, up2_flat, down2_flat = self._align_tensors(up1, down1, up2, down2)
                    
                    # Приводим к нужному dtype
                    up1_flat = up1_flat.to(device, dtype=dtype)
                    down1_flat = down1_flat.to(device, dtype=dtype)
                    up2_flat = up2_flat.to(device, dtype=dtype)
                    down2_flat = down2_flat.to(device, dtype=dtype)

                    # Смешиваем в зависимости от режима
                    if mode == "add":
                        up = up1_flat + up2_flat
                        down = down1_flat + down2_flat
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "weighted_avg":
                        up = (up1_flat + up2_flat) / 2
                        down = (down1_flat + down2_flat) / 2
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "weighted_sum":
                        # Нормализованное суммирование
                        weight1 = 0.5
                        weight2 = 0.5
                        up = up1_flat * weight1 + up2_flat * weight2
                        down = down1_flat * weight1 + down2_flat * weight2
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "interpolate":
                        t = 0.5
                        up = up1_flat * (1 - t) + up2_flat * t
                        down = down1_flat * (1 - t) + down2_flat * t
                        alpha = (alpha1 + alpha2) / 2

                    elif mode == "magnitude":
                        up_abs1 = torch.abs(up1_flat)
                        up_abs2 = torch.abs(up2_flat)
                        up = torch.where(up_abs1 > up_abs2, up1_flat, up2_flat)
                        
                        down_abs1 = torch.abs(down1_flat)
                        down_abs2 = torch.abs(down2_flat)
                        down = torch.where(down_abs1 > down_abs2, down1_flat, down2_flat)
                        alpha = alpha1 if alpha1 > alpha2 else alpha2

                    elif mode == "difference":
                        diff_up = up2_flat - up1_flat
                        diff_down = down2_flat - down1_flat
                        threshold_diff = 0.1
                        mask_up = torch.abs(diff_up) > threshold_diff
                        mask_down = torch.abs(diff_down) > threshold_diff
                        up = up1_flat + diff_up * mask_up.float()
                        down = down1_flat + diff_down * mask_down.float()
                        alpha = alpha1

                    elif mode == "concat":
                        # Конкатенация - увеличиваем ранг
                        up = torch.cat([up1_flat, up2_flat], dim=1)
                        down = torch.cat([down1_flat, down2_flat], dim=0)
                        alpha = alpha1 + alpha2

                    elif mode == "svd":
                        up, down = self._svd_merge(up1_flat, down1_flat, up2_flat, down2_flat, rank, threshold, device)
                        alpha = torch.tensor(rank, dtype=torch.int64)

                    else:
                        print(f"⚠️ Unknown mode: {mode}, using add")
                        up = up1_flat + up2_flat
                        down = down1_flat + down2_flat
                        alpha = (alpha1 + alpha2) / 2

                    # Восстанавливаем форму если была 4D
                    if orig_shape1 is not None and len(orig_shape1) == 4:
                        try:
                            up = up.view(orig_shape1)
                        except:
                            # Если не удалось восстановить, оставляем как есть
                            pass
                        try:
                            down = down.view(orig_shape1)
                        except:
                            pass
                    elif orig_shape2 is not None and len(orig_shape2) == 4:
                        try:
                            up = up.view(orig_shape2)
                        except:
                            pass
                        try:
                            down = down.view(orig_shape2)
                        except:
                            pass

                # Применяем output_scale
                up = up * output_scale
                down = down * output_scale

                # Записываем результат
                if suffix1 == "up/down":
                    weight[key + ".lora_up.weight"] = up.to(device, dtype=dtype)
                    weight[key + ".lora_down.weight"] = down.to(device, dtype=dtype)
                else:
                    weight[key + ".lora_B.weight"] = up.to(device, dtype=dtype)
                    weight[key + ".lora_A.weight"] = down.to(device, dtype=dtype)
                
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
        
        # Если ранги разные, приводим к одному
        if r1 != r2:
            if r1 < r2:
                # Дополняем до большего ранга
                pad_size = r2 - r1
                up1 = torch.nn.functional.pad(up1, (0, pad_size))
                down1 = torch.nn.functional.pad(down1, (0, pad_size))
                r1 = r2
            else:
                # Обрезаем до меньшего ранга
                up2 = up2[:, :r1]
                down2 = down2[:, :r1]
                r2 = r1
        
        # Сумма весов
        weight = (up1.view(-1, r1) @ down1.view(r1, -1)) + (up2.view(-1, r2) @ down2.view(r2, -1))
        weight = weight.to(torch.float32)
        
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        if threshold < 1.0:
            rank = self._index_sv_fro(S, threshold)
        rank = min(rank, len(S))
        
        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)
        Vh = Vh[:rank, :]
        
        if down1.dim() == 4:
            try:
                U = U.reshape(up1.shape[0], rank, 1, 1)
                Vh = Vh.reshape(rank, down1.shape[1], down1.shape[2], down1.shape[3])
            except:
                pass
            
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