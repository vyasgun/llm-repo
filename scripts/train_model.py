import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

def main(output_dir, dataset_path, model_id=None):
    """
    Main function to perform fine-tuning of a language model on Apple Silicon (M3).
    """
    # --- 1. Set the Device for M3 Macs ---
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Load the Base Model and Tokenizer ---
    # Use smaller model for CI/CD, full model for local development
    if model_id is None:
        # Check if running in CI (GitHub Actions)
        import os
        is_ci = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
        
        if is_ci:
            # Use smaller model for CI to avoid memory issues
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print("🔧 CI environment detected: Using TinyLlama for memory efficiency")
        else:
            # Use full model for local development
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            print("💻 Local environment: Using Meta-Llama-3-8B-Instruct")
    
    print(f"Loading model: {model_id}")

    # You CANNOT use BitsAndBytesConfig on M3, so load in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device}  # Map model to the specified device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Model loaded in full precision to the MPS device.")

    # --- 3. Prepare the Dataset ---
    print("\n--- Step 3: Preparing the Dataset ---")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    def format_data(example):
        # Use different templates based on model
        if "llama" in model_id.lower():
            # Llama 3 chat template
            text = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"You are a helpful assistant. Provide clear and concise answers.\n"
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{example['instruction']}\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"{example['output']}<|eot_id|>"
            )
        else:
            # Generic/TinyLlama chat template
            text = (
                f"<|system|>\nYou are a helpful assistant. Provide clear and concise answers.</s>\n"
                f"<|user|>\n{example['instruction']}</s>\n"
                f"<|assistant|>\n{example['output']}</s>"
            )
        return {"text": text}

    formatted_dataset = dataset.map(format_data, remove_columns=["instruction", "input", "output"])
    formatted_dataset = formatted_dataset.select(range(min(500, len(formatted_dataset))))

    print(f"Dataset formatted with {model_id} chat template. Sample:")
    print(formatted_dataset[0]["text"])

    # --- 4. Configure PEFT (LoRA) ---
    print("\n--- Step 4: Configuring and Applying PEFT (LoRA) ---")

    # Auto-detect target modules for LoRA
    def find_target_modules(model):
        """Automatically find Linear and Embedding modules for LoRA"""
        import torch.nn as nn
        target_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                names = name.split('.')
                target_modules.add(names[-1])
        return list(target_modules)

    # Get target modules automatically
    target_modules = find_target_modules(model)
    print(f"Auto-detected target modules: {target_modules}")
    
    # Filter to common attention/MLP modules if available
    common_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", 
                     "query", "key", "value", "dense", "fc1", "fc2", "out_proj", "c_attn", "c_proj"]
    filtered_targets = [t for t in target_modules if t in common_targets]
    
    # Use filtered targets if available, otherwise use first few detected modules
    final_targets = filtered_targets if filtered_targets else target_modules[:4]
    print(f"Using target modules: {final_targets}")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=final_targets
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 5. Start the Training Process ---
    print("\n--- Step 5: Starting the Training Process ---")
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        num_train_epochs=1,
        bf16=False,
        fp16=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        dataset_text_field="text",  # TRL handles tokenization automatically
        max_seq_length=1024,        # Use this to truncate long sequences
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    print("\nTraining complete!")

    # --- 6. Save and Merge the Fine-Tuned Model ---
    print("\n--- Step 6: Saving and Merging the Model ---")
    trainer.save_model(output_dir)

    # Clean up memory before merging
    del model, trainer
    torch.cuda.empty_cache()

    base_model_full = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    peft_model_to_merge = PeftModel.from_pretrained(
        base_model_full,
        output_dir,
        is_trainable=False
    )
    merged_model = peft_model_to_merge.merge_and_unload()
    merged_tokenizer = AutoTokenizer.from_pretrained(model_id)

    merged_model.save_pretrained(f"{output_dir}-merged")
    merged_tokenizer.save_pretrained(f"{output_dir}-merged")
    print("\nFine-tuned model successfully merged and saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Script")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument("--dataset_path", type=str, default="tatsu-lab/alpaca", help="Path or name of the training dataset.")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID to use. If not specified, auto-selects based on environment.")
    args = parser.parse_args()
    main(args.output_dir, args.dataset_path, args.model_id)