import comfy
import folder_paths
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
            
        # Создаем путь для сохранения
        lora_folder = folder_paths.folder_names_and_paths["loras"][0][0]
        save_path = os.path.join(lora_folder, f"{file_name}.{extension}")
        
        # Создаем папку если её нет
        os.makedirs(lora_folder, exist_ok=True)
        
        try:
            print(f"💾 Saving LoRA to: {save_path}")
            print(f"  • Total keys: {len(lora_data)}")
            
            # Сохраняем ВСЕГДА как есть, без масштабирования
            # Веса уже масштабированы в LoraLoaderWeightOnly или LoraMerger
            # strength_model и strength_clip игнорируем при сохранении
            new_state_dict = lora_data
            
            # Проверяем размер перед сохранением
            total_size = sum(v.numel() * v.element_size() for v in new_state_dict.values() if hasattr(v, 'numel'))
            print(f"  • Approximate size: {total_size / 1024 / 1024:.2f} MB")
            
            # Сохраняем файл
            comfy.utils.save_torch_file(new_state_dict, save_path)
            
            # Проверяем что файл создан
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"✅ LoRA saved successfully! Size: {file_size / 1024:.2f} KB")
                if file_size < 1024:
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