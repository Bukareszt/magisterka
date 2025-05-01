import os
import torch
import numpy as np
from tqdm import tqdm
import random
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def extract_first_round_prompt(example):
    """Extract the first round prompt"""
    conversation = example['conversation']
    user_content = ''
    
    # Combining the sentences from the first-round of the user prompt
    for i, sentence in enumerate(conversation):
        if sentence['role'] == 'user':
            if i > 0:
                user_content += '\n'
            user_content += sentence['content']
        else:
            break
    
    return user_content

def generate_response_and_get_length(prompts, model, tokenizer, device, max_new_tokens=512, batch_size=32):
    """Generate responses for prompts and return their token lengths"""
    output_lengths = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Process in batches to avoid OOM issues
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding for deterministic outputs
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Calculate response lengths (only count new tokens)
        for j, (prompt_ids, output_ids) in enumerate(zip(input_ids, outputs)):
            prompt_len = len(prompt_ids)
            response_len = len(output_ids) - prompt_len
            output_lengths.append(response_len)
            
    return output_lengths

def generate_output_length_data(data_size=100000, batch_size=1000, 
                                seed=42, inference_model=None, 
                                max_new_tokens=512, output_dir="./data"):
    """
    Generate and save the dataset for output length prediction
    
    Args:
        data_size: Number of samples to use
        batch_size: Process this many examples at a time to save memory
        seed: Random seed
        inference_model: Model to use for generating responses
        max_new_tokens: Maximum new tokens to generate
        output_dir: Directory to save the generated data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load inference model and tokenizer
    print(f"Loading inference model: {inference_model}")
    tokenizer = AutoTokenizer.from_pretrained(inference_model, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(inference_model, trust_remote_code=True).to(device)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset in streaming mode to save memory
    print(f"Loading lmsys/lmsys-chat-1m dataset in streaming mode (size: {data_size})...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    dataset = dataset.take(data_size)
    
    # Process dataset in batches to avoid memory issues
    all_prompts = []
    
    print("Extracting prompts from conversations...")
    batch_count = 0
    current_batch = []
    
    for example in tqdm(dataset, desc="Processing examples"):
        current_batch.append(example)
        
        # Process batch when it reaches the specified size
        if len(current_batch) >= batch_size:
            batch_count += 1
            print(f"Processing batch {batch_count}...")
            
            # Extract prompts
            batch_prompts = [extract_first_round_prompt(ex) for ex in current_batch]
            all_prompts.extend(batch_prompts)
            
            # Clear the batch
            current_batch = []
    
    # Process any remaining examples
    if current_batch:
        batch_prompts = [extract_first_round_prompt(ex) for ex in current_batch]
        all_prompts.extend(batch_prompts)
    
    print(f"Total examples processed: {len(all_prompts)}")
    
    # Apply random seed before splitting
    indices = list(range(len(all_prompts)))
    random.seed(seed)
    random.shuffle(indices)
    
    all_prompts = [all_prompts[i] for i in indices]
    
    # Generate responses and get output lengths using the inference model
    print("Generating responses to calculate output lengths...")
    all_lengths = generate_response_and_get_length(
        all_prompts, model, tokenizer, device, max_new_tokens=max_new_tokens, batch_size=batch_size
    )
    
    # Split into train and validation sets
    split_idx = int(len(all_prompts) * 0.9)  # 10% validation
    train_prompts = all_prompts[:split_idx]
    val_prompts = all_prompts[split_idx:]
    train_lengths = all_lengths[:split_idx]
    val_lengths = all_lengths[split_idx:]
    
    # Save the data
    train_data = {
        "prompts": train_prompts,
        "output_lengths": train_lengths
    }
    
    val_data = {
        "prompts": val_prompts,
        "output_lengths": val_lengths
    }
    
    print(f"Saving {len(train_prompts)} training samples and {len(val_prompts)} validation samples...")
    
    # Save as JSON
    with open(os.path.join(output_dir, f"output_len_train_{data_size}_{seed}.json"), "w") as f:
        json.dump(train_data, f)
    
    with open(os.path.join(output_dir, f"output_len_val_{data_size}_{seed}.json"), "w") as f:
        json.dump(val_data, f)
    
    print(f"Dataset generation complete. Files saved to {output_dir}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate data for output length prediction")
    parser.add_argument("--data_size", type=int, default=100000, help="Number of samples to use from dataset")
    parser.add_argument("--processing_batch_size", type=int, default=1000, 
                        help="Batch size for dataset processing (memory optimization)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--inference_model", type=str, required=True,
                       help="Model to use for generating responses (to create labels)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate for each response")
    parser.add_argument("--output_dir", type=str, default="./data", 
                        help="Directory to save the generated data")
    
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Generate data
    generate_output_length_data(
        data_size=args.data_size,
        batch_size=args.processing_batch_size,
        seed=args.seed,
        inference_model=args.inference_model,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 