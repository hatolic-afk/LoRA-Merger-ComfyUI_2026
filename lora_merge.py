"""
Ядро для корректного смешивания LoRA.

Правильная математика: суммируем полные матрицы весов, потом SVD.
Никаких cross-членов → никакого шума.
"""

import math
import torch
import comfy.utils


def _safe_svd(matrix, rank=None, use_rsvd=True, n_iter=2, oversampling=10):
    """SVD с опциональным рандомизированным ускорением."""
    m, n = matrix.shape
    max_rank = min(m, n)
    if max_rank <= 0:
        return None, None, None

    if rank is None or rank > max_rank:
        rank = max_rank
    rank = max(1, int(rank))

    oversampling = max(1, int(oversampling))
    max_oversampling = max_rank - rank
    if max_oversampling > 0:
        oversampling = min(oversampling, max_oversampling)
    oversampling = max(1, oversampling)

    try:
        if use_rsvd and rank < max_rank // 2 and (rank + oversampling) < n:
            Q = torch.randn(n, rank + oversampling,
                            device=matrix.device, dtype=matrix.dtype)
            Y = matrix @ Q
            for _ in range(max(0, int(n_iter))):
                Y = matrix @ (matrix.T @ Y)
            Q, _ = torch.linalg.qr(Y, mode='reduced')
            B = Q.T @ matrix
            U, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Q @ U
        else:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    except Exception as e:
        print(f"❌ SVD failed: {e}")
        return None, None, None

    rank = min(rank, len(S))
    return (U[:, :rank].contiguous(),
            S[:rank].contiguous(),
            Vh[:rank, :].contiguous())


def _detect_suffix(lora_data):
    for k in lora_data:
        if ".lora_up" in k:
            return "up/down"
        if ".lora_B" in k:
            return "A/B"
    return None


def _get_base_keys(lora_data, suffix):
    keys = set()
    for k in lora_data:
        if suffix == "up/down":
            idx = k.find(".lora_up")
            if idx >= 0:
                keys.add(k[:idx])
                continue
            idx = k.find(".lora_down")
            if idx >= 0:
                keys.add(k[:idx])
        else:
            idx = k.find(".lora_B")
            if idx >= 0:
                keys.add(k[:idx])
                continue
            idx = k.find(".lora_A")
            if idx >= 0:
                keys.add(k[:idx])
    return keys


def merge_multiple_loras(loras, weights, target_rank=16, device='cuda',
                         dtype='float32', use_rsvd=True,
                         rsvd_oversampling=10, rsvd_n_iter=2,
                         output_power=1.0):
    """Сливает список лор в одну.
    
    Args:
        loras: список словарей LoRA
        weights: список весов (обычно все 1.0, т.к. веса уже в лорах)
        target_rank: целевой ранг
        output_power: множитель для результата (0.0-2.0)
    """
    valid_loras = []
    valid_weights = []
    for lora, w in zip(loras, weights):
        if lora is None:
            continue
        lora_data = lora.get("lora", {}) if isinstance(lora, dict) else {}
        if not lora_data:
            continue
        valid_loras.append(lora_data)
        valid_weights.append(float(w))

    if not valid_loras:
        return {"lora": {}}

    if len(valid_loras) == 1:
        lora_data = valid_loras[0]
        if abs(output_power - 1.0) > 1e-10:
            result = {}
            for k, v in lora_data.items():
                if k.endswith(".alpha"):
                    result[k] = v
                else:
                    result[k] = v * output_power
            return {"lora": result}
        return {"lora": lora_data}

    suffix = None
    for lora_data in valid_loras:
        suffix = _detect_suffix(lora_data)
        if suffix:
            break
    if suffix is None:
        return {"lora": {}}

    all_keys = set()
    for lora_data in valid_loras:
        all_keys.update(_get_base_keys(lora_data, suffix))

    if not all_keys:
        return {"lora": {}}

    dtype_map = {"float32": torch.float32,
                 "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    target_dtype = dtype_map.get(dtype, torch.float32)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"🔀 Merging {len(valid_loras)} LoRAs, {len(all_keys)} keys, rank={target_rank}")

    weight_dict = {}
    pbar = comfy.utils.ProgressBar(len(all_keys))

    for key in all_keys:
        try:
            if suffix == "up/down":
                up_suf = ".lora_up.weight"
                down_suf = ".lora_down.weight"
            else:
                up_suf = ".lora_B.weight"
                down_suf = ".lora_A.weight"
            alpha_suf = ".alpha"

            up_k = key + up_suf
            down_k = key + down_suf
            alpha_k = key + alpha_suf

            W_sum = None
            original_shape = None
            is_conv = False

            for lora_data, w in zip(valid_loras, valid_weights):
                if up_k not in lora_data or down_k not in lora_data:
                    continue

                up = lora_data[up_k]
                down = lora_data[down_k]
                alpha = lora_data.get(alpha_k, torch.tensor(down.shape[0]))
                r = down.shape[0]

                if original_shape is None:
                    original_shape = up.shape
                    is_conv = (up.dim() > 2) or (down.dim() > 2)

                # Масштаб: w * alpha / rank
                if isinstance(alpha, torch.Tensor):
                    alpha_val = float(alpha.item())
                else:
                    alpha_val = float(alpha)
                scale = float(w) * (alpha_val / r) if r > 0 else 0.0

                if abs(scale) < 1e-12:
                    continue

                up_f = up.float().to(device=device) * scale
                down_f = down.float().to(device=device)
                up_flat = up_f.reshape(up_f.shape[0], -1)
                down_flat = down_f.reshape(down_f.shape[0], -1)
                W_contrib = up_flat @ down_flat

                if W_sum is None:
                    W_sum = W_contrib
                else:
                    W_sum = W_sum + W_contrib

            if W_sum is None:
                continue

            W_sum = torch.nan_to_num(W_sum, nan=0.0, posinf=0.0, neginf=0.0)
            if W_sum.abs().max() < 1e-12:
                continue

            max_rank = min(W_sum.shape[0], W_sum.shape[1])
            target = max(1, min(int(target_rank), max_rank))

            U, S, Vh = _safe_svd(W_sum, rank=target, use_rsvd=use_rsvd,
                                 n_iter=rsvd_n_iter, oversampling=rsvd_oversampling)
            if U is None:
                continue

            sqrt_S = torch.sqrt(S.clamp(min=0))
            up_merged = U * sqrt_S.unsqueeze(0)
            down_merged = sqrt_S.unsqueeze(1) * Vh

            # Применяем output_power
            if abs(output_power - 1.0) > 1e-10:
                up_merged = up_merged * output_power
                down_merged = down_merged * output_power

            # Восстанавливаем форму для сверток
            if is_conv and original_shape is not None:
                if len(original_shape) == 4:
                    up_merged = up_merged.reshape(original_shape[0], target, 1, 1)
                    down_merged = down_merged.reshape(target, down_merged.shape[1] // (original_shape[2] * original_shape[3]) 
                                                      if down_merged.dim() == 2 else down_merged.shape[1], 
                                                      original_shape[2] if len(original_shape) >= 3 else 1,
                                                      original_shape[3] if len(original_shape) >= 4 else 1)
                elif len(original_shape) == 3:
                    up_merged = up_merged.reshape(original_shape[0], target, 1)
                    down_merged = down_merged.reshape(target, -1)

            if suffix == "up/down":
                weight_dict[key + ".lora_up.weight"] = up_merged.to(dtype=target_dtype).cpu().contiguous()
                weight_dict[key + ".lora_down.weight"] = down_merged.to(dtype=target_dtype).cpu().contiguous()
            else:
                weight_dict[key + ".lora_B.weight"] = up_merged.to(dtype=target_dtype).cpu().contiguous()
                weight_dict[key + ".lora_A.weight"] = down_merged.to(dtype=target_dtype).cpu().contiguous()

            weight_dict[key + ".alpha"] = torch.tensor(target, dtype=torch.int64, device='cpu')

        except Exception as e:
            print(f"❌ Error on {key}: {e}")
        pbar.update(1)

    if not weight_dict:
        return {"lora": {}}

    print(f"✅ Merged {len(weight_dict) // 3} modules")
    return {"lora": weight_dict}


class LoraMerger:
    """Слияние двух LoRA (для совместимости)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "master_lora": ("LoRA",),
                "lora_2":      ("LoRA",),
                "target_rank":  ("INT", {"default": 16, "min": 1, "max": 320, "step": 1}),
                "output_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "device":       (["cuda", "cpu"], {"default": "cuda"}),
                "dtype":        (["float32", "float16", "bfloat16"], {"default": "float32"}),
                "use_rsvd":     ("BOOLEAN", {"default": True}),
                "rsvd_oversampling": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "rsvd_n_iter":  ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "merge_loras"
    CATEGORY = "lora_merge"

    def merge_loras(self, master_lora, lora_2,
                    target_rank=16, output_power=1.0,
                    device="cuda", dtype="float32",
                    use_rsvd=True, rsvd_oversampling=10, rsvd_n_iter=2):
        result = merge_multiple_loras(
            [master_lora, lora_2], [1.0, 1.0],
            target_rank=target_rank, device=device, dtype=dtype,
            use_rsvd=use_rsvd, rsvd_oversampling=rsvd_oversampling,
            rsvd_n_iter=rsvd_n_iter, output_power=output_power)
        return (result,)

    @classmethod
    def IS_CHANGED(s, master_lora, lora_2, **kwargs):
        import hashlib
        key = f"{id(master_lora)}_{id(lora_2)}_" + "_".join(
            f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(key.encode()).hexdigest()