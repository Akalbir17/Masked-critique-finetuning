import os
import random
import torch
import warnings
import bitsandbytes as bnb
from datetime import datetime
from typing import Optional
from peft import PeftModel
from transformers import AutoTokenizer
import copy
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    set_seed,
    get_cosine_schedule_with_warmup
)

from btsft.func.format_reward import format_reward_func
from btsft.func.parameters import get_parameters_count
from btsft.func.mapping import map_iio, map_conversations
from btsft.trainers.blurred_thoughts import BlurredThoughtsSFTTrainer

from transformers.trainer_pt_utils import get_parameter_names
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["WANDB_API_KEY"] = "3f2439a07988bc703ef8aeaedf541dd194740f8c"


def train(
    model_name: str,
    checkpoint: str,
    threshold: float = 0.2,
    bf_beta: float = 0.05,
    lora_rank: int = 64,
    dataset_train: str = None,
    max_length: int = 512,
    batch_size: int = 32,
    accumulation_iter: int = 128,
    epochs: int = 1,
    lr: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    logging_dir: str = "./logs",
    output_dir: str = "./results",
    save_steps: int = 500,
    train_test_split: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    num_workers: int = 24,
    skip: int = 0,
    take: Optional[int] = None,
    tokenizer_name: Optional[str] = None,
    trainer_checkpoint: Optional[str] = None,
    response_template: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> None:
    # Set unsloth environment variable
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    print(f"Training {model_name} on {dataset_train} with {tokenizer_name} tokenizer.")

    set_seed(seed)

    if tokenizer_name is None:
        tokenizer_name = checkpoint

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = 16384 
    # 🔍 Debug: Check how <think> is tokenized
    tokens = tokenizer.encode("<think>", add_special_tokens=False)
    print("🔎 Token IDs for <think>:", tokens)
    print("🔎 Number of tokens:", len(tokens))

    tokens_close = tokenizer.encode("</think>", add_special_tokens=False)
    print("🔎 Token IDs for </think>:", tokens_close)
    print("🔎 Number of tokens:", len(tokens_close))


    # make sure tokenizer contains <think> and </think> tokens
    # if "<think>" not in tokenizer.special_tokens_map.values():
    #     print("Adding special tokens <think> and </think> to tokenizer.")
    #     tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    # Add all special tags used in the dataset
    special_tokens = ["<think>", "</think>",
                    "<critique>", "</critique>",
                    "<reasoning>", "</reasoning>",
                    "<ans>", "</ans>"]

    existing_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])

    # Only add tokens that don't already exist
    new_tokens = [tok for tok in special_tokens if tok not in existing_tokens]

    if new_tokens:
        print(f"Adding special tokens: {new_tokens}")
        tokenizer.add_special_tokens({"additional_special_tokens": existing_tokens + new_tokens})

    print("Loading model...")
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=checkpoint,
    #     load_in_4bit=True,
    #     max_lora_rank=lora_rank,
    #     cache_dir=cache_dir,
    # )

    local_dir = "agentica-org/DeepScaleR-1.5B-Preview"
    # Initialize tokenizer with chat template
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_dir,
        max_length=max_length,
        dtype=torch.bfloat16,
        load_in_4bit = False,
        load_in_8bit = False,
        # full_finetuning=True,
        # trust_remote_code=True,
        # fast_inference=True,
        use_exact_model_name= True
    )
    tokenizer.model_max_length = 16384  # Force again to be sure

    # Check model max positions
    print("Tokenizer max length:", tokenizer.model_max_length)
    print("Model max position embeddings:", model.config.max_position_embeddings)

    # If model doesn't support long context yet:
    if model.config.max_position_embeddings < 16384:
        print(f"🔧 Updating model max_position_embeddings from {model.config.max_position_embeddings} to 16384")
        model.config.max_position_embeddings = 16384

    model.gradient_checkpointing_enable()


    # resize token embeddings if necessary
    if model.config.vocab_size != len(tokenizer):
        print("Resizing token embeddings...")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=lora_rank,
    #     target_modules=[
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "o_proj",
    #         "gate_proj",
    #         "up_proj",
    #         "down_proj",
    #     ],
    #     lora_alpha=lora_rank,
    #     use_gradient_checkpointing="unsloth",
    #     random_state=seed,
    # )

    print("Model parameters:", get_parameters_count(model))
    # 📊 Diagnostic: Is this full fine-tune or LoRA?
    diagnostic_report(model, name=model_name)

    dataset_validation = None

    if dataset_train is not None:
        if dataset_train.endswith(".csv"):
            print("📄 Loading CSV dataset...")
            df = pd.read_csv(dataset_train)

            # Filter to keep only rows where answer_match is True
            df = df[df["answer_match"] == True]

            # Check required columns exist
            if "prompt" not in df.columns or "critique" not in df.columns:
                raise ValueError("CSV must contain 'prompt' and 'critique' columns after filtering.")

            # Convert to HuggingFace dataset
            datasets = Dataset.from_pandas(df)

            print("🛠 Detected CSV format: prompt + critique → conversations")

            # Define system instruction
            system_instruction = (
                "You are given with user prompt and language model output. "
                "Your task is to first critique the language model output and then use the useful reasoning chains "
                "in language model output to generate the final answer. Before generating answer think step by step "
                "to give your reasoning. The output format is given below: "
                "<critique> ... generated critique ...</critique>\n"
                "<reasoning> ... reasoning chains ... </reasoning>\n"
                "<ans> \\boxed{{51}} </ans>"
            )

            # Format response template
            response_template = (
                "<|start_header_id|>system<|end_header_id|>\n"
                + system_instruction
                + "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

            # def build_conversation(example):
            #     return {
            #         # "conversations": f"{example['prompt'].strip()}{response_template}<think>{example['critique'].strip()}</think>"
            #         "conversations": (
            #                             f"<|start_header_id|>system<|end_header_id|>\n"
            #                             f"{system_instruction}\n"
            #                             f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            #                             f"{example['prompt'].strip()}\n\n"
            #                             f"<think>{example['critique'].strip()}</think>"
            #                         )
            #     }
            def build_conversation(example):
                prompt = example['prompt'].strip()
                critique_and_reasoning = example['critique'].strip()

                # Try to split out the <ans> if already included inside critique
                if "<ans>" in critique_and_reasoning:
                    critique_reasoning_part = critique_and_reasoning.split("<ans>")[0].strip()
                    answer_part = "<ans>" + critique_and_reasoning.split("<ans>")[1].strip()
                else:
                    # If <ans> is not present separately, we just put a placeholder
                    critique_reasoning_part = critique_and_reasoning
                    answer_part = "<ans>\\boxed{UNKNOWN}</ans>"

                return {
                    "conversations": (
                        f"<|start_header_id|>system<|end_header_id|>\n"
                        f"{system_instruction}\n"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                        f"{prompt}\n\n"
                        f"<think>{critique_reasoning_part}</think>\n"
                        f"{answer_part}"
                    )
                }


            datasets = datasets.map(build_conversation)
            datasets = datasets.remove_columns([col for col in datasets.column_names if col != "conversations"])
            datasets = datasets.filter(lambda x: x["conversations"] is not None, num_proc=1)

            print("✅ Sample after formatting:")
            print(datasets[0])
            print("✅ Total examples after filtering:", len(datasets))


        # datasets = load_dataset(dataset_train, 
        #                         split="train",
        #                         cache_dir=cache_dir)

        # # 🔥 Add this print block
        # print("🔥 Loaded raw dataset:")
        # print("Type:", type(datasets))
        # print("Length:", len(datasets))
        # print("Keys in first item:", datasets[0].keys())
        # print("First item:", datasets[0])


        # # if "conversations" in datasets.column_names:
        # #     datasets = datasets.map(lambda x: {'conversations': map_conversations(x['conversations'])},
        # #                             num_proc=num_workers).filter(
        # #         lambda x: x["conversations"] is not None, num_proc=num_workers
        # #     )
        # # else:
        # #     datasets = datasets.map(
        # #         lambda x: {"conversations": map_iio(x)}, num_proc=num_workers
        # #     ).filter(lambda x: x["conversations"] is not None, num_proc=num_workers)


        # # if "question" in datasets.column_names and "answer" in datasets.column_names:
        # #     print("🛠 Detected LIMO format: question + answer → conversations")

        # #     response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n" if response_template is None else response_template

        # #     def build_conversation(example):
        # #         return {
        # #             "conversations": f"{example['question'].strip()}{response_template}{example['answer'].strip()}"
        # #         }
        # if "question" in datasets.column_names and "answer" in datasets.column_names and "solution" in datasets.column_names:
        #     print("🛠 Detected LIMO format: question + solution + answer → conversations")

        #     response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n" if response_template is None else response_template

        #     def build_conversation(example):
        #         return {
        #             "conversations": f"{example['question'].strip()}{response_template}<think>{example['solution'].strip()}</think><answer>{example['answer'].strip()}</answer>"
        #         }

        #     datasets = datasets.map(build_conversation, num_proc=num_workers)
        #     datasets = datasets.remove_columns([col for col in datasets.column_names if col != "conversations"])
        #     datasets = datasets.filter(lambda x: x["conversations"] is not None, num_proc=1)  # keep num_proc=1 for debug

        #     print("✅ Sample after formatting:")
        #     print(datasets[0])
        #     print("✅ Total examples after filtering:", len(datasets))


        elif "conversations" in datasets.column_names:
            datasets = datasets.map(lambda x: {'conversations': map_conversations(x['conversations'])},
                                    num_proc=num_workers).filter(
                lambda x: x["conversations"] is not None, num_proc=num_workers
            )

        else:
            datasets = datasets.map(
                lambda x: {"conversations": map_iio(x)}, num_proc=num_workers
            ).filter(lambda x: x["conversations"] is not None, num_proc=num_workers)

        datasets = [datasets]
    else:
        raise ValueError("Dataset not provided.")
    
    if len(datasets[0]) == 0:
        raise ValueError("No valid examples found in the dataset.")

    dataset = concatenate_datasets(datasets).shuffle(seed=seed)

    if take is not None:
        dataset = dataset.take(take)

    if not dataset_validation:
        dataset = dataset.train_test_split(test_size=train_test_split)

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n" if response_template is None else response_template

    # def tokenize_function(examples):
    #     """
    #     Tokenize the examples and mask tokens between <think> tags.
    #     This is the main function for tokenizing the dataset introducing Blurred Thoughts.
    #     Setting the label to -100 effectively masks the token for the model. 
    #     This prevents the model from strictly following the training data and encourages it to produce more diverse responses, aligned with its own probability distribution.

    #     Args:
    #         examples: Examples from the dataset

    #     Returns:
    #         Dictionary containing input_ids, attention_mask and labels
    #     """
    #     tokens = tokenizer(
    #         tokenizer.apply_chat_template(
    #             examples["conversations"], tokenize=False, add_generation_prompt=False
    #         ),
    #         max_length=max_length,
    #         padding="max_length",
    #         truncation=True,
    #         verbose=False,
    #         return_tensors="pt",
    #     )
    #     tokens["labels"] = tokens["input_ids"].clone()
    #     think_open_token = tokenizer.encode("<think>", add_special_tokens=False)[0]
    #     think_close_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
    #     response_template_len = len(
    #         tokenizer.encode(response_template, add_special_tokens=False)
    #     )

    #     masked_counter = 0

    #     for i, token in enumerate(tokens["labels"]):
    #         think_open = -1
    #         for j, y in enumerate(token):
    #             if y == think_open_token:
    #                 think_open = j
    #                 tokens["labels"][i][: j - response_template_len - 1] = -100
    #                 continue

    #             if y == think_close_token:
    #                 think_open = -1
    #                 break

    #             if think_open > 0 and think_open + 5 < j:
    #                 if random.random() < threshold:
    #                     tokens["labels"][i][j] = -100
    #                     masked_counter += 1

    #     tokens["input_ids"] = tokens["input_ids"].squeeze(0)
    #     tokens["attention_mask"] = tokens["attention_mask"].squeeze(0)
    #     tokens["labels"] = tokens["labels"].squeeze(0)
    #     # ⬇️ ADD THIS HERE (after squeezing)
    #     masked_tokens = (tokens["labels"] == -100).sum().item()
    #     total_tokens = tokens["labels"].numel()
    #     masking_ratio = masked_tokens / total_tokens

    #     if masking_ratio > 0:
    #         print(f"✅ Masked {masked_tokens} / {total_tokens} tokens ({masking_ratio:.2%})")
    #     else:
    #         print("❌ No masking detected.")

    #     return tokens
    def tokenize_function(examples):
        """
        Tokenize the examples and mask tokens between <think> tags.
        Skips apply_chat_template to preserve original tags.
        """

        # 👇 Skip apply_chat_template — use your formatted conversation as-is
        final_prompt = examples["conversations"]

        tokens = tokenizer(
            final_prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            verbose=False,
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()

        # 🔍 Decode the tokenized input so we can see exactly what the model sees
        decoded = tokenizer.decode(tokens["input_ids"][0])
        print("\n===== Decoded sequence before masking =====")
        print(decoded[:2000])  # First 2000 chars to avoid too much output
        print("==========================================\n")

        # Get the token IDs for <think> and </think>
        think_open_token = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_close_token = tokenizer.encode("</think>", add_special_tokens=False)[0]

        response_template_len = len(
            tokenizer.encode(response_template, add_special_tokens=False)
        )

        print("Response template length (in tokens):", response_template_len)

        masked_counter = 0

        for i, token_seq in enumerate(tokens["labels"]):
            think_open = -1
            for j, y in enumerate(token_seq):
                if y == think_open_token:
                    print(f"<think> found at position {j} in example {i}")
                    think_open = j

                    # Mask everything before <think>, except the response template
                    mask_until = max(0, j - response_template_len - 1)
                    tokens["labels"][i][:mask_until] = -100
                    continue

                if y == think_close_token:
                    print(f"</think> found at position {j} in example {i}")
                    think_open = -1
                    continue

                if think_open > 0 and j > think_open + 5:
                    # Force masking for debugging
                    tokens["labels"][i][j] = -100
                    masked_counter += 1

        tokens["input_ids"] = tokens["input_ids"].squeeze(0)
        tokens["attention_mask"] = tokens["attention_mask"].squeeze(0)
        tokens["labels"] = tokens["labels"].squeeze(0)

        masked_tokens = (tokens["labels"] == -100).sum().item()
        total_tokens = tokens["labels"].numel()
        masking_ratio = masked_tokens / total_tokens

        if masking_ratio > 0:
            print(
                f"✅ Masked {masked_tokens} / {total_tokens} tokens ({masking_ratio:.2%})"
            )
        else:
            print("❌ No masking detected.")

        return tokens



    items_to_skip = skip

    print("Tokenizing dataset...")
    if not dataset_validation:
        train_samples = dataset["train"].shape[0]
        test_samples = dataset["test"].shape[0]
        print(f"{train_samples} training samples, {test_samples} test samples")
        if trainer_checkpoint is None:
            train_ds = (
                dataset["train"]
                .skip(items_to_skip)
                .map(
                    tokenize_function,
                    num_proc=num_workers,
                    remove_columns=dataset["train"].column_names,
                )
            )
            test_ds = dataset["test"].map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )
        else:
            train_ds = dataset["train"].map(
                tokenize_function,
                num_proc=num_workers,
                batched=False,
                remove_columns=dataset["train"].column_names,
            )
            test_ds = dataset["test"].map(
                tokenize_function,
                num_proc=num_workers,
                batched=False,
                remove_columns=dataset["train"].column_names,
            )
        print(
            f"{dataset['train'].shape[0]} training samples, {dataset['test'].shape[0]} test samples"
        )
    else:
        if trainer_checkpoint is None:
            train_ds = dataset.map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )
            test_ds = dataset_validation.map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )
        else:
            train_ds = dataset.map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )
            test_ds = dataset_validation.map(
                tokenize_function,
                num_proc=num_workers,
                remove_columns=dataset["train"].column_names,
            )

        train_samples = dataset.shape[0] - items_to_skip
        print(
            f"{train_samples} training samples, {dataset_validation.dataset_size} test samples"
        )

    max_steps = train_samples // (batch_size * accumulation_iter) * epochs

    print(f"Batch size {batch_size}")

    output_dir = os.path.join(
        os.getcwd(),
        output_dir,
        model_name,
        datetime.now().strftime("%Y-%m-%d %H_%M_%S"),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n in decay_parameters
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    print("Setting up optimizer.")
    optimizer = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
        lr=float(lr),
    )

    print("Setting up scheduler.")
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        learning_rate=lr,
        max_grad_norm=1.0,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        save_strategy="steps",
        save_steps=save_steps,
        disable_tqdm=False,
        push_to_hub=False,
        logging_strategy="steps",
        logging_dir=os.path.join(
            os.getcwd(),
            logging_dir,
            model_name,
            datetime.now().strftime("%Y-%m-%d %H_%M_%S"),
        ),
        logging_steps=1,
        logging_nan_inf_filter=True,
        gradient_accumulation_steps=accumulation_iter,
        output_dir=output_dir,
        max_steps=max_steps,
        evaluation_strategy="steps",
        eval_steps=save_steps,
        eval_accumulation_steps=accumulation_iter,
        seed=seed,
        bf16=True,
        include_num_input_tokens_seen=True,
        save_safetensors=True,
        neftune_noise_alpha=0.1,
        split_batches=True,
        save_total_limit=1,
        use_cpu=device == "cpu",
    )

    if not hasattr(model, "base_model"):
        model.base_model = model

    class Dummy:
        def forward(self, *args, **kwargs):
            return model.forward(*args, **kwargs)

    model.base_model.model = Dummy()
    
    print("Training model.")
    
    trainer = BlurredThoughtsSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.with_format("torch"),
        eval_dataset=test_ds.with_format("torch"),
        data_collator=collator,
        optimizers=(optimizer, lr_scheduler),
        tokenizer=tokenizer,
        bf_beta=bf_beta,
        format_reward_func=format_reward_func,
    )

    trainer.train(resume_from_checkpoint=trainer_checkpoint)

    print("Training complete.")
    trainer.save_model(os.path.join(output_dir, "final"))
    
    # After trainer.save_model(...)
    print("✅ Merging LoRA into base model...")



    # Merge
    # model = PeftModel.from_pretrained(model, os.path.join(output_dir, "final"))
    # model = model.merge_and_unload()  # merges and removes LoRA
    # Just save model and tokenizer as is
    model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)

    # 🔧 Resize embeddings again just in case (safe)
    model.resize_token_embeddings(len(tokenizer))

    # Save
    merged_output_dir = os.path.join(output_dir, "merged")
    print(f"💾 Saving to {merged_output_dir}")
    model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)
    print("✅ Merged model saved successfully!")



    
import os
from datetime import datetime

def diagnostic_report(model, save_to_file: bool = True, name: str = "model_report"):
    print("\n=== 🔍 Full Finetune Diagnostic Report ===")

    from peft import PeftModel
    is_lora = isinstance(model, PeftModel)

    result = []
    result.append(f"🧪 LoRA/PEFT Active?: {'✅ Yes (LoRA)' if is_lora else '❌ No (Full Fine-Tune)'}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent_trainable = 100 * trainable_params / total_params
    result.append(f"🧠 Total Parameters:     {total_params:,}")
    result.append(f"🔧 Trainable Parameters: {trainable_params:,}")
    result.append(f"📊 % Trainable:          {percent_trainable:.2f}%")

    result.append("\n📂 Sample Trainable Layers:")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            result.append(f"  ✅ {name} | dtype: {param.dtype} | shape: {tuple(param.shape)}")
            count += 1
            if count >= 5:
                break
    if count == 0:
        result.append("  ❌ No trainable layers found!")

    # Dtype and device
    try:
        result.append(f"\n🔬 Model dtype: {model.dtype}")
    except:
        result.append("🔬 Could not access model.dtype directly.")

    device = next(model.parameters()).device
    result.append(f"💻 Device: {device}")
    result.append("==========================================\n")

    print("\n".join(result))

    if save_to_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "./diagnostics"
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{name}_{timestamp}.txt")
        with open(file_path, "w") as f:
            f.write("\n".join(result))
        print(f"📝 Saved diagnostic to: {file_path}")
