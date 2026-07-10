# LoRA Merge Suite для ComfyUI

Набор нодов для загрузки, мержа и сохранения LoRA весов с поддержкой 8 методов объединения.

## Установка
cd ComfyUI/custom_nodes/
git clone https://github.com/hatolic-afk/LoRA-Merger-ComfyUI_2026.git

## Список нодов

### 1. LoraLoaderWeightOnly
Загружает LoRA из файла, применяет LBW и масштабирует веса.

Входные параметры:
- lora_name (STRING) - Имя файла LoRA из папки loras
- strength_model (FLOAT, default: 1.0) - Сила применения к модели (-20.0 до 20.0)
- strength_clip (FLOAT, default: 1.0) - Сила применения к CLIP (-20.0 до 20.0)
- lbw (STRING, default: "") - Блочные веса для тонкой настройки

Выход:
- LoRA - Загруженные и масштабированные веса

### 2. LoraMerger
Объединяет две LoRA в одну.

Входные параметры:
- master_lora (LoRA) - Основная LoRA
- lora_2 (LoRA, optional) - Вторая LoRA
- mode (STRING, default: "add") - Метод мержа: add, concat, svd, weighted_avg, weighted_sum, interpolate, magnitude, difference
- rank (INT, default: 16) - Ранг для SVD (1-320)
- threshold (FLOAT, default: 1.0) - Порог для SVD (0-1)
- device (STRING, default: "cuda") - Устройство: cuda/cpu
- dtype (STRING, default: "float32") - Тип данных: float32/float16/bfloat16
- output_scale (FLOAT, default: 1.0) - Масштаб результата (0.0-2.0)

Выход:
- LoRA - Объединенная LoRA

### 3. LoraMergerStack
Объединяет до 10 LoRA последовательно.

Входные параметры:
- master_lora (LoRA) - Основная LoRA
- lora_2 до lora_10 (LoRA, optional) - Дополнительные LoRA
- mode (STRING, default: "add") - Метод мержа
- rank (INT, default: 16) - Ранг для SVD
- threshold (FLOAT, default: 1.0) - Порог для SVD
- device (STRING, default: "cuda") - Устройство
- dtype (STRING, default: "float32") - Тип данных
- output_scale (FLOAT, default: 1.0) - Масштаб результата

Выход:
- LoRA - Объединенная LoRA

### 4. LoraLoaderFromWeight
Применяет LoRA к модели и CLIP.

Входные параметры:
- model (MODEL) - Модель для применения
- clip (CLIP) - CLIP для применения
- lora (LoRA) - LoRA веса

Выход:
- MODEL - Модель с примененной LoRA
- CLIP - CLIP с примененной LoRA

### 5. LoraSaveToFile
Сохраняет LoRA в файл.

Входные параметры:
- lora (LoRA) - LoRA для сохранения
- file_name (STRING, default: "merged") - Имя файла без расширения
- extension (STRING, default: "safetensors") - Расширение файла

Выход: Нет (OUTPUT_NODE)

## Методы мержа (mode)

ADD - Суммирует веса (0.8 + 0.6 = 1.4). Стандартный метод для объединения стилей.
WEIGHTED_AVG - Суммирует веса (как ADD), т.к. веса уже масштабированы через strength.
WEIGHTED_SUM - Суммирует и нормализует, чтобы веса не взрывались.
INTERPOLATE - Сферическая интерполяция 50/50 для плавного перехода.
MAGNITUDE - Берет максимальные значения, выделяя доминирующий стиль.
DIFFERENCE - Добавляет только значимые различия (порог 0.1).
CONCAT - Объединяет ранги, сохраняя всю информацию.
SVD - Сжимает через сингулярное разложение, уменьшая размер.

## Сравнение методов

ADD: Вся инфа, артефакты могут быть, быстро, просто
WEIGHTED_AVG: Вся инфа, артефактов мин, быстро, просто
WEIGHTED_SUM: Вся инфа, артефактов мин, быстро, средне
INTERPOLATE: Вся инфа, артефактов нет, средне, сложно
MAGNITUDE: Сильная инфа, артефакты могут быть, быстро, просто
DIFFERENCE: Новая инфа, артефактов нет, быстро, средне
CONCAT: Вся инфа, артефактов нет, средне, средне
SVD: Сжатая инфа, артефактов мин, медленно, очень сложно

## Схема работы

LoraLoaderWeightOnly -> загружает safetensors -> применяет LBW -> масштабирует по strength -> возвращает ТОЛЬКО ВЕСА

LoraMerger -> берет две LoRA -> мержит по методу -> возвращает ТОЛЬКО ВЕСА

LoraMergerStack -> берет до 10 LoRA -> мержит последовательно -> возвращает ТОЛЬКО ВЕСА

LoraLoaderFromWeight -> берет LoRA -> применяет к модели с силой 1.0 (веса уже масштабированы)

LoraSaveToFile -> берет LoRA -> сохраняет как safetensors

## Примеры использования

Пример 1: Простой мерж двух лор
LoraLoaderWeightOnly (lora1.safetensors, strength=0.8) -> LoRA1
LoraLoaderWeightOnly (lora2.safetensors, strength=0.6) -> LoRA2
LoraMerger (master=LoRA1, lora_2=LoRA2, mode="add") -> Merged
LoraLoaderFromWeight (model, clip, Merged) -> (model, clip)

Пример 2: Мерж с усреднением
LoraLoaderWeightOnly (style.safetensors, strength=1.0) -> LoRA1
LoraLoaderWeightOnly (detail.safetensors, strength=1.0) -> LoRA2
LoraMerger (master=LoRA1, lora_2=LoRA2, mode="weighted_avg") -> Merged
LoraLoaderFromWeight (model, clip, Merged) -> (model, clip)

Пример 3: Стек из 5 лор
LoraLoaderWeightOnly (lora1.safetensors, strength=0.7) -> L1
LoraLoaderWeightOnly (lora2.safetensors, strength=0.5) -> L2
LoraLoaderWeightOnly (lora3.safetensors, strength=0.8) -> L3
LoraLoaderWeightOnly (lora4.safetensors, strength=0.6) -> L4
LoraLoaderWeightOnly (lora5.safetensors, strength=0.4) -> L5
LoraMergerStack (master_lora=L1, lora_2=L2, lora_3=L3, lora_4=L4, lora_5=L5, mode="add") -> Merged
LoraLoaderFromWeight (model, clip, Merged) -> (model, clip)

Пример 4: Сохранение результата
LoraLoaderWeightOnly (lora1.safetensors, strength=0.8) -> L1
LoraLoaderWeightOnly (lora2.safetensors, strength=0.6) -> L2
LoraMerger (master=L1, lora_2=L2, mode="add") -> Merged
LoraSaveToFile (lora=Merged, file_name="my_merged_lora") -> сохраняет my_merged_lora.safetensors

## LBW (Layer Block Weights)

Позволяет задавать разные веса для разных блоков модели.

Формат: block1, block2, block3, ...

Примеры:
1.0, 0.8, 0.6, 0.4, 0.2 - убывающие веса
0.0, 1.0, 0.0, 1.0, 0.0 - выборочные блоки

Можно использовать пресеты из файла preset.txt

## Важные моменты

1. Все веса масштабируются в LoraLoaderWeightOnly через strength_model и strength_clip
2. LoraLoaderFromWeight применяет LoRA с силой 1.0 (веса уже масштабированы)
3. LoraMerger работает с уже масштабированными весами - не нужно дополнительных параметров
4. Формат LoRA везде одинаковый: {"lora": {weights}} - без лишних метаданных
5. Поддерживаются два формата: lora_up/lora_down и lora_A/lora_B

## Лицензия

MIT

## Вклад

PR и Issue приветствуются!
