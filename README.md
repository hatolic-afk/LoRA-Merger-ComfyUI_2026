# LoRA Merge Suite for ComfyUI

Набор нод для корректного слияния LoRA моделей без шума.

## Установка
Скопировать папку в `ComfyUI/custom_nodes/` и перезапустить ComfyUI.

## Ноды

### LoraLoaderWeightOnly
Загружает LoRA с весом и LBW.

**Входы:**
- `lora_name` - имя файла
- `weight` - вес (-20.0 … 20.0)
- `lbw` - per-block веса ("0.8, 0.4, 0.0, 1.0")

**Выходы:**
- `LoRA` - загруженная лора

---

### LoraMergerMulti
Слияние до 5 LoRA.

**Входы:**
- `lora_1` … `lora_5` - LoRA для слияния
- `target_rank` - целевой ранг (1-320)
- `output_power` - множитель (0.0-2.0)
- `device` - cuda/cpu
- `dtype` - float32/float16/bfloat16
- `use_rsvd` - использовать RSVD
- `rsvd_oversampling` - (1-50)
- `rsvd_n_iter` - (0-10)

**Выходы:**
- `LoRA` - слитая лора

---

### LoraMerger
Слияние 2 LoRA.

**Входы:**
- `master_lora` - первая лора
- `lora_2` - вторая лора
- `target_rank` - целевой ранг
- `output_power` - множитель
- `device` - cuda/cpu
- `dtype` - float32/float16/bfloat16
- `use_rsvd` - использовать RSVD
- `rsvd_oversampling` - (1-50)
- `rsvd_n_iter` - (0-10)

**Выходы:**
- `LoRA` - слитая лора

---

### LoraLoaderFromWeight
Применяет LoRA к модели и CLIP.

**Входы:**
- `model` - модель
- `clip` - CLIP
- `lora` - LoRA
- `strength_model` - сила (-20.0 … 20.0)
- `strength_clip` - сила (-20.0 … 20.0)

**Выходы:**
- `MODEL`, `CLIP`

---

### LoraSaveToFile
Сохраняет LoRA в файл.

**Входы:**
- `lora` - LoRA
- `file_name` - имя файла
- `extension` - safetensors

**Выходы:**
- нет (Output Node)

## Пример
LoraLoaderWeightOnly (lora_1, weight=1.0) → LoRA_1
LoraLoaderWeightOnly (lora_2, weight=0.8) → LoRA_2
LoraLoaderWeightOnly (lora_3, weight=1.2) → LoRA_3

LoraMergerMulti (
lora_1=LoRA_1,
lora_2=LoRA_2,
lora_3=LoRA_3,
target_rank=16,
output_power=1.0
) → Merged_LoRA

LoraLoaderFromWeight (model, clip, lora=Merged_LoRA) → MODEL, CLIP
LoraSaveToFile (lora=Merged_LoRA, file_name="merged") → сохранение


## Математика

**Проблема:** старое слияние создавало шум.
up = up1 + up2
down = down1 + down2
W = (up1+up2) @ (down1+down2)
W = up1@down1 + up1@down2 + up2@down1 + up2@down2


`up1@down2` и `up2@down1` — это шум.

**Решение:** правильное суммирование матриц.
W_merged = Σ((alpha_i / rank_i) * (up_i @ down_i))
U, S, V = svd(W_merged)
up_new = U * sqrt(S)
down_new = sqrt(S) * V


**Результат:** без шума, даже при смешивании 5 лор.

## Параметры

| Параметр | Описание | Рекомендация |
|----------|----------|--------------|
| `target_rank` | ранг после сжатия | 16-32 |
| `output_power` | масштаб результата | 1.0 |
| `use_rsvd` | Randomized SVD | True |
