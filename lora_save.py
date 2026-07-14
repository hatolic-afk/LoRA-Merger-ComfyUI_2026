import os
import folder_paths
import comfy.utils


class LoraSaveToFile:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora":      ("LoRA",),
                "file_name": ("STRING", {"multiline": False, "default": "merged"}),
                "extension": (["safetensors"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_lora_to_file"
    CATEGORY = "lora_merge"
    OUTPUT_NODE = True

    def save_lora_to_file(self, lora, file_name, extension):
        if not lora:
            print("❌ No LoRA data")
            return {}

        lora_data = lora.get("lora", {}) if isinstance(lora, dict) else {}
        if not lora_data:
            print("❌ Empty LoRA data")
            return {}

        lora_folder = folder_paths.folder_names_and_paths["loras"][0][0]
        save_path = os.path.join(lora_folder, f"{file_name}.{extension}")
        os.makedirs(lora_folder, exist_ok=True)

        try:
            print(f"💾 Saving LoRA to: {save_path}")
            comfy.utils.save_torch_file(lora_data, save_path)
            print(f"✅ LoRA saved! Size: {os.path.getsize(save_path) / 1024:.2f} KB")
        except Exception as e:
            print(f"❌ Error saving LoRA: {e}")

        return {}

    @classmethod
    def IS_CHANGED(s, lora, file_name, extension):
        import time
        return time.time()