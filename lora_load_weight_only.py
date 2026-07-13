import comfy
import folder_paths
import os
import re
import math

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PRESET_FILE = os.path.join(CURRENT_DIR, "preset.txt")

def extract_numbers(s):
    return [int(num) for num in re.findall(r'\d+', s)]

def expand_lbw(weight_list):
    length = len(weight_list)
    if length == 17:
        new_list = []
        j = 0
        for i in range(26):
            if i in [2, 5, 8, 11, 12, 13, 15, 16, 17]:
                new_list.append(0.0)
            else:
                if j < len(weight_list):
                    new_list.append(weight_list[j])
                    j += 1
                else:
                    new_list.append(0.0)
    elif length == 12:
        new_list = []
        j = 0
        for i in range(20):
            if i in [2, 3, 4, 5, 8, 18, 19, 20]:
                new_list.append(0.0)
            else:
                if j < len(weight_list):
                    new_list.append(weight_list[j])
                    j += 1
                else:
                    new_list.append(0.0)
    else:
        new_list = weight_list
    return new_list

def parse_weight_preset(text):
    lines = text.strip().split("\n")
    weight_dict = {}
    for line in lines:
        if ":" in line:
            key, values = line.split(":", 1)
            float_values = [float(x.strip()) for x in values.split(",") if x.strip()]
            weight_dict[key.strip()] = float_values
    return weight_dict

def parse_weight_list(text):
    if os.path.exists(PRESET_FILE):
        try:
            with open(PRESET_FILE, "r") as f:
                dic = parse_weight_preset(f.read())
        except Exception as e:
            print(f"Error reading preset file: {e}")
            dic = {}
    else:
        dic = {}

    if text in dic:
        return dic[text]
    else:
        cleaned_text = text.replace(" ", "")
        if cleaned_text:
            return [float(weight) for weight in cleaned_text.split(",") if weight]
        else:
            return []

MID_ID = {26: 13, 20: 10}

class LoraLoaderWeightOnly:
    def __init__(self):
        self.loaded_lora = None
        self.lbw = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lbw": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ("LoRA", )
    FUNCTION = "load_lora_weight_only"
    CATEGORY = "lora_merge"

    def load_lora_weight_only(self, lora_name, strength_model, strength_clip, lbw):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None or self.lbw != lbw:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                print(f"📂 Loaded {lora_name}: {len(lora)} keys")
            except Exception as e:
                print(f"❌ Error loading LoRA {lora_name}: {e}")
                return ({"lora": {}}, )

            if lbw != "":
                weight_list = parse_weight_list(lbw)
                if weight_list:
                    print(f"  • Applying LBW: {weight_list}")
                    weight_list = expand_lbw(weight_list)
                    length = len(weight_list)
                    
                    # Определяем тип LoRA
                    up_keys = [key for key in lora.keys() if "lora_up" in key and not "lora_te" in key]
                    if not up_keys:
                        up_keys = [key for key in lora.keys() if "lora_B" in key and not "lora_te" in key]
                    
                    # Группируем ключи по блокам
                    block_keys = {}
                    for key in up_keys:
                        ids = extract_numbers(key)
                        block_id = 0
                        
                        if "input_blocks" in key:
                            block_id = ids[0] if ids else 0
                        elif "middle_block" in key:
                            block_id = MID_ID.get(length, 13)
                        elif "output_blocks" in key:
                            block_id = ids[0] + MID_ID.get(length, 13) + 1 if ids else 0
                        elif "down_blocks" in key:
                            if len(ids) >= 2:
                                block_id = ids[0]*3 + ids[1] + 1
                            else:
                                block_id = 0
                            if "down_sampler" in key:
                                block_id += 2
                        elif "mid_block" in key:
                            block_id = MID_ID.get(length, 13)
                        elif "up_blocks" in key:
                            if len(ids) >= 2:
                                block_id = ids[0]*3 + ids[1] + MID_ID.get(length, 13) + 1
                            else:
                                block_id = 0
                            if "up_sampler" in key:
                                block_id += 2
                        else:
                            block_id = 0
                        
                        if block_id < len(weight_list):
                            if block_id not in block_keys:
                                block_keys[block_id] = []
                            block_keys[block_id].append(key)
                    
                    # Применяем веса к блокам
                    for block_id, keys in block_keys.items():
                        weight = weight_list[block_id]
                        if weight != 0.0:
                            # Масштабируем все ключи блока одинаково
                            sqrt_w = math.sqrt(abs(weight))
                            sign_w = 1 if weight >= 0 else -1
                            scale = sqrt_w * sign_w
                            
                            for key in keys:
                                # Масштабируем UP
                                lora[key] = lora[key] * scale
                                
                                # Находим и масштабируем DOWN
                                down_key = key.replace("lora_up", "lora_down").replace("lora_B", "lora_A")
                                if down_key in lora:
                                    lora[down_key] = lora[down_key] * scale
                        else:
                            # Удаляем блок
                            for key in keys:
                                if key in lora:
                                    del lora[key]
                                down_key = key.replace("lora_up", "lora_down").replace("lora_B", "lora_A")
                                if down_key in lora:
                                    del lora[down_key]
                                alpha_key = key.replace("lora_up.weight", "alpha").replace("lora_B.weight", "alpha")
                                if alpha_key in lora:
                                    del lora[alpha_key]

            # Применяем глобальное масштабирование
            if strength_model != 1.0 or strength_clip != 1.0:
                print(f"  • Scaling weights: model={strength_model}, clip={strength_clip}")
                
                # Группируем ключи по типу
                model_keys = []
                clip_keys = []
                for key in lora.keys():
                    if "lora_te" in key:
                        clip_keys.append(key)
                    elif "lora_up" in key or "lora_down" in key or "lora_A" in key or "lora_B" in key:
                        model_keys.append(key)
                
                # Масштабируем модель
                if strength_model != 1.0:
                    sqrt_model = math.sqrt(abs(strength_model))
                    sign_model = 1 if strength_model >= 0 else -1
                    scale_model = sqrt_model * sign_model
                    for key in model_keys:
                        lora[key] = lora[key] * scale_model
                
                # Масштабируем CLIP
                if strength_clip != 1.0:
                    sqrt_clip = math.sqrt(abs(strength_clip))
                    sign_clip = 1 if strength_clip >= 0 else -1
                    scale_clip = sqrt_clip * sign_clip
                    for key in clip_keys:
                        lora[key] = lora[key] * scale_clip

            self.loaded_lora = (lora_path, lora)
            self.lbw = lbw

        return ({"lora": lora}, )

    @classmethod
    def IS_CHANGED(s, lora_name, strength_model, strength_clip, lbw):
        import hashlib
        key = f"{lora_name}_{strength_model}_{strength_clip}_{lbw}"
        return hashlib.md5(key.encode()).hexdigest()