import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from datasets import load_dataset, Dataset
import argparse
import random
from logger import Logger  # Import the Logger class
from vicuna_bert import VicunaToBertRegressor

class OutputLengthDataset(Dataset):
    def __init__(self, prompts, output_lengths):
        self.prompts = prompts
        self.output_lengths = output_lengths
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        # Handle case where idx might be a list or other iterable
        if isinstance(idx, (list, tuple, np.ndarray)):
            return {
                'prompt': [self.prompts[i] for i in idx],
                'output_length': [self.output_lengths[i] for i in idx]
            }
        # Normal case - idx is an integer
        return {
            'prompt': self.prompts[idx],
            'output_length': self.output_lengths[idx]
        }

def extract_first_round_prompt(example, vicuna_tokenizer):
    """Extract the first round prompt and response length"""
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
    
    # Combining the sentences from the first-round of the assistant response
    assistant_content = ''
    for j in range(i, len(conversation)):
        sentence = conversation[j]
        if sentence['role'] == 'assistant':
            if j > i:
                assistant_content += '\n'
            assistant_content += conversation[j]['content']
        else:
            break

    # Add specified number of words from the response if requested
    prompt = user_content
        
    # Calculate output length (number of tokens)
    encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
    output_length = len(encoded_response['input_ids'])
    
    return prompt, output_length

def prepare_lmsys_dataset(data_size=100000, model_name="vicuna-13b", first_round_only=False, seed=42):
    """
    Load and prepare the lmsys-chat-1m dataset for output length prediction
    
    Args:
        data_size: Number of samples to use
        model_name: Name of the model to filter for
        first_round_only: Whether to use only the first round of conversation
        seed: Random seed
        
    Returns:
        train_prompts, val_prompts, train_lengths, val_lengths
    """
    # Initialize tokenizer for measuring output lengths
    vicuna_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", use_fast=False)
    
    # Load dataset
    print(f"Loading lmsys/lmsys-chat-1m dataset (size: {data_size})...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    dataset = dataset.select(range(data_size))
    
    # Filter for the specified model and shuffle the dataset
    print(f"Filtering for model: {model_name}")
    filtered_dataset = dataset.filter(lambda example: example["model"] == model_name)
    filtered_dataset = filtered_dataset.shuffle(seed=seed)
    
    print(f"Dataset filtered: {len(filtered_dataset)} samples")
    
    def process_example(example):
        prompt, output_len = extract_first_round_prompt(example, vicuna_tokenizer)
        return {"prompt": prompt, "output_length": output_len}

    # Process the dataset
    processed_dataset = filtered_dataset.map(process_example)
    processed_dataset = processed_dataset.filter(lambda x: x["output_length"] > 1)
    
    # Extract prompts and output lengths as lists
    print("Processing conversations...")
    prompts = processed_dataset["prompt"]
    output_lengths = processed_dataset["output_length"]
    
    # Split into train and validation sets
    train_prompts, val_prompts, train_lengths, val_lengths = train_test_split(
        prompts, output_lengths, test_size=0.1, random_state=seed
    )
    
    # Convert to lists to ensure compatibility
    train_prompts = list(train_prompts)
    val_prompts = list(val_prompts)
    train_lengths = list(train_lengths) 
    val_lengths = list(val_lengths)
    
    print(f"Dataset prepared: {len(train_prompts)} training samples, {len(val_prompts)} validation samples")
    return train_prompts, val_prompts, train_lengths, val_lengths

def collate_fn(batch):
    """Custom collate function for dataloader"""
    prompts = [item['prompt'] for item in batch]
    output_lengths = torch.tensor([item['output_length'] for item in batch], dtype=torch.float)
    return prompts, output_lengths

def train(
    model,
    train_dataloader,
    val_dataloader,
    device,
    num_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=0,
    output_dir="./saved_models",
    n_tokens=1,
    patience=3,
    logger=None,  # Add logger parameter
):
    """Training loop for the model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Track best validation loss for early stopping and model saving
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        train_progress_bar = tqdm(train_dataloader, desc="Training")
        for prompts, output_lengths in train_progress_bar:
            output_lengths = output_lengths.to(device)
            
            # Forward pass
            predictions = model(prompts, n_tokens=n_tokens)
            
            # Calculate loss
            loss = criterion(predictions, output_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            train_progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / train_steps
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_mae = 0.0  # Mean Absolute Error for more interpretable metric
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc="Validation")
            for prompts, output_lengths in val_progress_bar:
                output_lengths = output_lengths.to(device)
                
                # Forward pass
                predictions = model(prompts, n_tokens=n_tokens)
                
                # Calculate loss
                loss = criterion(predictions, output_lengths)
                val_loss += loss.item()
                
                # Calculate MAE
                mae = torch.abs(predictions - output_lengths).mean().item()
                val_mae += mae
                
                val_steps += 1
                val_progress_bar.set_postfix({"val_loss": loss.item(), "val_mae": mae})
        
        avg_val_loss = val_loss / val_steps
        avg_val_mae = val_mae / val_steps
        print(f"Validation loss: {avg_val_loss:.4f}, Validation MAE: {avg_val_mae:.2f} tokens")
        
        # Log metrics to W&B if logger is provided
        if logger:
            metrics = {
                "loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_mae": avg_val_mae,
            }
            logger.log_metrics(metrics, step=epoch, prefix="train")
        
        # Save best model and check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save model
            model_save_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
            model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_mae': avg_val_mae,
            }
            torch.save(model_state, model_save_path)
            print(f"Model saved to {model_save_path}")
            
            # Log model checkpoint to W&B if logger is provided
            if logger and logger.log_model:
                logger.log_model_checkpoint(model, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered. Training stopped after completing {epoch+1} epochs")
                break
    
    return model

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train output length prediction model")
    parser.add_argument("--data_size", type=int, default=100000, help="Number of samples to use from dataset")
    parser.add_argument("--model_name", type=str, default="vicuna-13b", help="Model name to filter in dataset")
    parser.add_argument("--first_round_only", action="store_true", default=True, help="Use only first round of conversation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for scheduler")
    parser.add_argument("--vicuna_model", type=str, default="lmsys/vicuna-7b-v1.3", help="Vicuna model for prediction")
    parser.add_argument("--bert_model", type=str, default="prajjwal1/bert-tiny", help="BERT model for regression")
    parser.add_argument("--output_dir", type=str, default="./saved_models/output_length_predictor", help="Directory to save models")
    parser.add_argument("--n_tokens", type=int, default=1, help="Number of tokens to generate for prediction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Add W&B arguments
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, help="W&B project name", default="output-length-prediction")
    parser.add_argument("--log_model", action="store_true", help="Whether to log model checkpoints to W&B")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("Loading and preparing dataset...")
    train_prompts, val_prompts, train_lengths, val_lengths = prepare_lmsys_dataset(
        data_size=args.data_size,
        model_name='vicuna-13b',
        first_round_only=args.first_round_only,
        seed=args.seed
    )
    
    # Ensure prompts and lengths are proper lists
    train_prompts = list(train_prompts)
    val_prompts = list(val_prompts)
    train_lengths = list(train_lengths)
    val_lengths = list(val_lengths)
    
    train_dataset = OutputLengthDataset(train_prompts, train_lengths)
    val_dataset = OutputLengthDataset(val_prompts, val_lengths)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print(f"Initializing model with {args.vicuna_model} and {args.bert_model}")
    model = VicunaToBertRegressor(vicuna_name=args.vicuna_model, bert_name=args.bert_model)
    device = model.device  # Use the device determined in the model initialization
    
    # Calculate warmup steps
    warmup_steps = int(len(train_dataloader) * args.num_epochs * args.warmup_ratio)
    
    # Initialize logger if W&B is enabled
    logger = None
    if args.use_wandb:
        config = {
            'task_type': 'regression',
            'model_name': args.model_name,
            'first_round_only': args.first_round_only,
            'data_size': args.data_size,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'warmup_ratio': args.warmup_ratio,
            'vicuna_model': args.vicuna_model,
            'bert_model': args.bert_model,
            'n_tokens': args.n_tokens,
            'seed': args.seed,
        }
        logger = Logger(
            config=config,
            model_name=args.model_name,
            project_name=args.wandb_project,
            enable_logging=True,
            log_model=args.log_model
        )
    
    # Train model
    print("Starting training...")
    model = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        output_dir=args.output_dir,
        n_tokens=args.n_tokens,
        logger=logger  # Pass logger to train function
    )
    
    # Final evaluation
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_steps = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for prompts, output_lengths in tqdm(val_dataloader, desc="Final Evaluation"):
            output_lengths = output_lengths.to(device)
            predictions = model(prompts, n_tokens=args.n_tokens)
            loss = criterion(predictions, output_lengths)
            val_loss += loss.item()
            mae = torch.abs(predictions - output_lengths).mean().item()
            val_mae += mae
            val_steps += 1
    
    avg_val_loss = val_loss / val_steps
    avg_val_mae = val_mae / val_steps
    
    print(f"Final validation - Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.2f} tokens")
    
    # Log final metrics
    if logger:
        final_metrics = {
            "val_loss": avg_val_loss,
            "val_mae": avg_val_mae
        }
        logger.log_metrics(final_metrics, prefix="final")
        logger.finish()
    
    print("Training complete!")

if __name__ == "__main__":
    main()
    