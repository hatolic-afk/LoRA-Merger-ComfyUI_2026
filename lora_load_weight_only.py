import comfy
import folder_paths
import os
import re

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
            if i in LBW17TO26:
                new_list.append(0.0)
            else:
                new_list.append(weight_list[j])
                j += 1
    elif length == 12:
        new_list = []
        j = 0
        for i in range(20):
            if i in LBW12TO20:
                new_list.append(0.0)
            else:
                new_list.append(weight_list[j])
                j += 1
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

LBW17TO26 = [2, 5, 8, 11, 12, 13, 15, 16, 17]
LBW12TO20 = [2, 3, 4, 5, 8, 18, 19, 20]
MID_ID = {26:13, 20:10}

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

        # Проверяем кеш
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        # Загружаем LoRA если нужно
        if lora is None or self.lbw != lbw:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                print(f"📂 Loaded {lora_name}: {len(lora)} keys")
            except Exception as e:
                print(f"❌ Error loading LoRA {lora_name}: {e}")
                return ({"lora": {}, "strength_model": strength_model, "strength_clip": strength_clip}, )
            
            # Применяем LBW если указано
            if lbw != "":
                weight_list = parse_weight_list(lbw)
                if weight_list:
                    print(f"  • Applying LBW: {weight_list}")
                    weight_list = expand_lbw(weight_list)
                    length = len(weight_list)

                    strength_clip = strength_clip * weight_list[0] if len(weight_list) > 0 else strength_clip

                    up_keys = [key for key in lora.keys() if "lora_up" in key and not "lora_te" in key]
                    keys_to_delete = []
                    
                    for key in up_keys:
                        ids = extract_numbers(key)
                        if "input_blocks" in key:
                            block_id = ids[0] if ids else 0
                        elif "middle_block" in key:
                            block_id = MID_ID.get(length, 13)
                        elif "output_blocks" in key:
                            block_id = ids[0] + MID_ID.get(length, 13) + 1 if ids else 0
                        elif "down_blocks" in key:
                            block_id = ids[0]*3 + ids[1] + 1 if len(ids) >= 2 else 0
                            if "down_sampler" in key:
                                block_id += 2
                        elif "mid_block" in key:
                            block_id = MID_ID.get(length, 13)
                        elif "up_blocks" in key:
                            block_id = ids[0]*3 + ids[1] + MID_ID.get(length, 13) + 1 if len(ids) >= 2 else 0
                            if "up_sampler" in key:
                                block_id += 2
                        else:
                            block_id = 0
                        
                        if block_id < len(weight_list):
                            weight = weight_list[block_id]
                            if weight != 0.0:
                                lora[key] = lora[key] * weight
                            else:
                                keys_to_delete.append(key)
                                down_key = key.replace("lora_up", "lora_down")
                                if down_key in lora:
                                    keys_to_delete.append(down_key)
                                alpha_key = key.replace("lora_up.weight", "alpha")
                                if alpha_key in lora:
                                    keys_to_delete.append(alpha_key)
                    
                    for key in set(keys_to_delete):
                        if key in lora:
                            del lora[key]
            
            self.loaded_lora = (lora_path, lora)
            self.lbw = lbw

        # Возвращаем структуру LoRA
        return ({"lora": lora, "strength_model": strength_model, "strength_clip": strength_clip}, )

    @classmethod
    def IS_CHANGED(s, lora_name, strength_model, strength_clip, lbw):
        import hashlib
        key = f"{lora_name}_{strength_model}_{strength_clip}_{lbw}"
        return hashlib.md5(key.encode()).hexdigest()