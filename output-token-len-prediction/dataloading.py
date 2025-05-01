import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

def generate_dataloaders(dataset, train_batch_size, test_batch_size, tokenizer, 
                          flag_first_round_only=True, task_type=0, num_classes=3):
    """
    Generate DataLoaders for training, validation, and testing.
    
    Args:
        dataset: The dataset to split and load.
        train_batch_size: Batch size for training and validation.
        test_batch_size: Batch size for testing.
        tokenizer: Tokenizer for padding/truncation.
        flag_first_round_only: Whether to use only first round data.
        task_type: Type of task (0=regression, 1-2=classification, etc.).
        num_classes: Number of classes for classification tasks.
        
    Returns:
        train_dataloader, validation_dataloader, test_dataset, weights
    """
    # Use streaming mode for large datasets to avoid memory issues
    dataset = dataset.with_format("torch")
    n_total_samples = len(dataset)
    if flag_first_round_only:
        # Use streaming train_test_split for better memory efficiency
        train_validationtest = dataset.train_test_split(test_size=0.4, shuffle=False, load_from_cache_file=False)
        validation_test = train_validationtest['test'].train_test_split(test_size=0.5, shuffle=False, load_from_cache_file=False)
        train_dataset = train_validationtest['train']
        validation_dataset = validation_test['train']
        test_dataset = validation_test['test']
    else:
        sep_train_val = int(n_total_samples * 0.6)
        sep_val_test = int(n_total_samples * 0.8)
        # Make sure that sentences from the same conversation would not appear across train/val/test:
        while sep_train_val < sep_val_test and abs(dataset[sep_train_val]['conversation_id'] - dataset[sep_train_val - 1]['conversation_id']) < 0.1:
            sep_train_val += 1
        while sep_val_test < n_total_samples and abs(dataset[sep_val_test]['conversation_id'] - dataset[sep_val_test - 1]['conversation_id']) < 0.1:
            sep_val_test += 1
        print('Total training samples: ', sep_train_val)
        print('Total validation samples: ', sep_val_test - sep_train_val)
        print('Total test samples: ', n_total_samples - sep_val_test)

        # Use iterable-style splits for memory efficiency
        train_dataset = dataset.select(range(sep_train_val))
        validation_dataset = dataset.select(range(sep_train_val, sep_val_test))
        test_dataset = dataset.select(range(sep_val_test, n_total_samples))
        train_dataset = train_dataset.shuffle(seed=1)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create DataLoaders with memory-efficient approach
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False, 
        batch_size=train_batch_size, 
        collate_fn=data_collator,
        pin_memory=True,  # Speed up data transfer to GPU
        num_workers=2     # Parallel data loading
    )
    
    validation_dataloader = DataLoader(
        validation_dataset, 
        shuffle=True, 
        batch_size=train_batch_size, 
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=2
    )
    
    weights = []
    if task_type == 1 or task_type == 2:
        # More memory-efficient way to count class examples
        class_counts = [0] * num_classes
        
        # Count in batches to avoid loading everything at once
        for batch in DataLoader(dataset, batch_size=512, collate_fn=lambda x: x):
            for item in batch:
                label = item["labels"]
                if 0 <= label < num_classes:
                    class_counts[label] += 1
                    
        for i in range(num_classes):
            print(f'Number of samples for class {i}: {class_counts[i]}')
            if class_counts[i] == 0:
                weights.append(0.0)
            else:
                weights.append(1.0 / class_counts[i])
    
    return train_dataloader, validation_dataloader, test_dataset, weights 