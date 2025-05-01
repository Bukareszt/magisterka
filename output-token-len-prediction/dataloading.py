import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

def generate_dataloaders(dataset, train_batch_size, test_batch_size, tokenizer, 
                        flag_first_round_only=True, task_type=0, num_classes=5, seed=42):
    """
    Generate train, validation, and test dataloaders from the dataset.
    
    Args:
        dataset: The input dataset
        train_batch_size: Batch size for training
        test_batch_size: Batch size for testing
        tokenizer: Tokenizer to use
        flag_first_round_only: Whether using only first round data
        task_type: Type of task (0=regression, 1=binary cls, 2=multi-cls)
        num_classes: Number of classes for classification
        seed: Random seed for reproducibility
        
    Returns:
        train_dataloader, validation_dataloader, test_dataset, weights
    """
    # Split the dataset into train, validation, and test sets with fixed seed
    splits = dataset.train_test_split(test_size=0.2, seed=seed)
    train_dataset = splits['train']
    test_splits = splits['test'].train_test_split(test_size=0.5, seed=seed)
    validation_dataset = test_splits['train']
    test_dataset = test_splits['test']

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # For classification tasks, calculate class weights
    weights = None
    if task_type == 1 or task_type == 2:
        # Calculate class weights for the loss function based on training data
        label_counts = [0] * num_classes
        for sample in train_dataset:
            label_counts[sample['labels']] += 1
        
        total_samples = sum(label_counts)
        weights = [total_samples / (len(label_counts) * count) if count > 0 else 0 for count in label_counts]
        print(f"Class weights based on label distribution: {weights}")

    # Create data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create dataloaders with fixed seeds for shuffling
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=train_batch_size, 
        collate_fn=data_collator,
        generator=torch.Generator().manual_seed(seed)  # Set generator with seed
    )
    
    validation_dataloader = DataLoader(
        validation_dataset, 
        shuffle=False, 
        batch_size=test_batch_size, 
        collate_fn=data_collator
    )

    return train_dataloader, validation_dataloader, test_dataset, weights 