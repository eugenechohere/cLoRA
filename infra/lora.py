import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, concatenate_datasets

from liger_kernel.transformers import AutoLigerKernelForCausalLM



def train_lora(data_path: str, output_path: str, gpu_id: int = 0, base_adapter_path: str = None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoLigerKernelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    base_model.gradient_checkpointing_enable()
    
    if base_adapter_path and os.path.exists(base_adapter_path):
        print(f"Loading existing adapter from {base_adapter_path}")
        model = PeftModel.from_pretrained(base_model, base_adapter_path, is_trainable=True)
        print("Continuing training from existing adapter")
    else:
        print("Starting fresh LoRA training")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
    
    # Ensure LoRA parameters require gradients
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    
    model.print_trainable_parameters()
    
    # dataset = load_dataset("json", data_files=data_path, split="train")

    # Load both datasets
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def format_qa_pair(examples):
        """Convert Q&A pairs to Qwen chat format"""
        texts = []
        for question, answer in zip(examples["prompt"], examples["completion"]):
            # Using Qwen's chat template format
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(
        format_qa_pair,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=2048,
            padding=False,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    # TODO: finalize the design
    tokenized_dataset = tokenized_dataset.select(range(640))

    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        # warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={
            "min_lr": 1e-4,
        },
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        logging_steps=1,
        # save_steps=100,
        # save_total_limit=2,
        # save_strategy="steps",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        torch_compile=False,
        optim="adamw_torch_fused",
        report_to="none",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        group_by_length=True,
        # lr_scheduler_type='constant',
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    # SJ: this only saves the PEFT weights (v small)
    model.save_pretrained(output_path)
    print(f"LoRA adapter saved to {output_path}")

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) >= 4:
    #     train_lora(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1, 
    #                sys.argv[4] if len(sys.argv) > 4 else None)
    # else:
    #     train_lora(sys.argv[1], sys.argv[2])

    
    # # 
    train_lora(
        'data/batch.jsonl',
        output_path='lora_output',
        gpu_id=1,
        base_adapter_path=None,
    )