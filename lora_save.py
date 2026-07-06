import comfy
import folder_paths
import math
import os

class LoraSaveToFile:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "lora": ("LoRA",),
                              "file_name": ("STRING", {"multiline": False, "default": "merged"}),
                              "extension": (["safetensors"], ),
                              }}
    RETURN_TYPES = ()
    FUNCTION = "save_lora_to_file"
    CATEGORY = "lora_merge"
    OUTPUT_NODE = True

    def save_lora_to_file(self, lora, file_name, extension):
        # Проверяем входные данные
        if not lora:
            print("❌ Error: No LoRA data to save")
            return {}
            
        lora_data = lora.get("lora", {})
        if not lora_data:
            print("❌ Error: Empty LoRA data!")
            print(f"  lora keys: {list(lora.keys())}")
            return {}
            
        strength_model = lora.get("strength_model", 1.0)
        strength_clip = lora.get("strength_clip", 1.0)
        
        # Создаем путь для сохранения
        lora_folder = folder_paths.folder_names_and_paths["loras"][0][0]
        save_path = os.path.join(lora_folder, f"{file_name}.{extension}")
        
        # Создаем папку если её нет
        os.makedirs(lora_folder, exist_ok=True)
        
        try:
            print(f"💾 Saving LoRA to: {save_path}")
            print(f"  • Total keys: {len(lora_data)}")
            print(f"  • Strength model: {strength_model}")
            print(f"  • Strength clip: {strength_clip}")
            
            # Применяем масштабирование если нужно
            if strength_model == 1.0 and strength_clip == 1.0:
                new_state_dict = lora_data
                print("  • No scaling applied (both strengths = 1.0)")
            else:
                new_state_dict = {}
                scaled_count = 0
                for key, value in lora_data.items():
                    if "lora_te" in key:
                        scale = strength_clip
                    else:
                        scale = strength_model
                        
                    if scale != 1.0:
                        sqrt_scale = math.sqrt(abs(scale))
                        sign_scale = 1 if scale >= 0 else -1
                        
                        if "lora_up" in key:
                            new_state_dict[key] = value * sqrt_scale * sign_scale
                            scaled_count += 1
                        elif "lora_down" in key:
                            new_state_dict[key] = value * sqrt_scale
                            scaled_count += 1
                        else:
                            new_state_dict[key] = value
                    else:
                        new_state_dict[key] = value
                
                print(f"  • Scaled {scaled_count} keys")
            
            # Проверяем размер перед сохранением
            total_size = sum(v.numel() * v.element_size() for v in new_state_dict.values() if hasattr(v, 'numel'))
            print(f"  • Approximate size: {total_size / 1024 / 1024:.2f} MB")
            
            # Сохраняем файл
            comfy.utils.save_torch_file(new_state_dict, save_path)
            
            # Проверяем что файл создан
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"✅ LoRA saved successfully! Size: {file_size / 1024:.2f} KB")
                if file_size < 1024:  # Меньше 1KB
                    print(f"⚠️ WARNING: File is very small ({file_size} bytes)! Something might be wrong.")
            else:
                print(f"❌ Error: File was not created!")
            
        except Exception as e:
            print(f"❌ Error saving LoRA: {e}")
            import traceback
            traceback.print_exc()
            raise e

        return {}

    @classmethod
    def IS_CHANGED(s, lora, file_name, extension):
        import time
        return time.time()