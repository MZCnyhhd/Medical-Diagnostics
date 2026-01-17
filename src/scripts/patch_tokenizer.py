
import os
import shutil

def patch_file():
    target_path = r"E:\LocalModels\Hugging Face\HuatuoGPT-7B\tokenization_baichuan.py"
    print(f"Patching: {target_path}")
    
    if not os.path.exists(target_path):
        print("❌ File not found!")
        return

    with open(target_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # The buggy pattern: super().__init__ called BEFORE sp_model initialization
    # We want to find the block and reorder it.
    
    # We will look for the specific chunks and replace them.
    # Since the file content is known, we can be specific.
    
    old_block = """        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)"""

    new_block = """        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )"""

    if old_block in content:
        new_content = content.replace(old_block, new_block)
        # Backup first
        shutil.copy2(target_path, target_path + ".bak")
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ File patched successfully!")
    elif new_block in content:
        print("⚠️ File is already patched.")
    else:
        print("❌ Could not find the exact code block to patch. Please check the file manually.")
        # Debug: print snippet
        start_idx = content.find("super().__init__")
        if start_idx != -1:
            print("Found super().__init__ at:", start_idx)
            print("Context:\n", content[start_idx:start_idx+500])

def clear_cache():
    # Cache path from traceback: C:\Users\MZCny\.cache\huggingface\modules\transformers_modules\HuatuoGPT_hyphen_7B
    cache_path = r"C:\Users\MZCny\.cache\huggingface\modules\transformers_modules\HuatuoGPT_hyphen_7B"
    print(f"\nClearing cache at: {cache_path}")
    
    if os.path.exists(cache_path):
        try:
            shutil.rmtree(cache_path)
            print("✅ Cache cleared successfully!")
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
    else:
        print("⚠️ Cache directory not found (already cleared or never existed).")

if __name__ == "__main__":
    patch_file()
    clear_cache()
