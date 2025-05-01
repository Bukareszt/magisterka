import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import evaluate
import matplotlib.pyplot as plt

def predict(model, dataloader, device, flag_vicuna_data_only=False, 
            flag_first_round_only=True, task_type=0, model_names=None):
    """
    Generate predictions for a dataset.
    
    Args:
        model: Model to use for predictions.
        dataloader: DataLoader for the dataset to predict on.
        device: Device to run predictions on.
        flag_vicuna_data_only: Whether using only vicuna data.
        flag_first_round_only: Whether using only first round data.
        task_type: Type of task (0=regression, 1=classification, etc.).
        model_names: List of model names for multi-model evaluation.
        
    Returns:
        DataFrame containing predictions and related information.
    """
    model.eval()
    predicted_labels = []
    actual_lengths = []
    latencies = []
    print_model_names = []
    turn_ids = []
    with torch.no_grad():
        for batch in dataloader:
            start_time = time.time()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if not flag_vicuna_data_only:
                model_ids = np.argmax(batch['model'].numpy(), axis=-1)
            if flag_vicuna_data_only:
                predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_name = batch['model'].to(device)
                predictions = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
            if task_type == 0 or task_type == 3 or task_type == 4:
                lengths = batch['num_tokens']
                predictions = predictions
            else:
                predictions = torch.argmax(predictions, dim=-1)
                lengths = batch['num_tokens']
            end_time = time.time()

            predicted_labels.extend(predictions.cpu().numpy())
            actual_lengths.extend(lengths.numpy())
            if not flag_first_round_only:
                turn_ids.extend(batch['turn_id'].numpy())
            latencies.append(end_time - start_time)
            for sample_i in range(len(input_ids)):
                if flag_vicuna_data_only:
                    print_model_names.append('vicuna-13b')
                else:
                    print_model_names.append(model_names[model_ids[sample_i]])

    if flag_first_round_only:
        df = pd.DataFrame({
            'actual_length': actual_lengths, 
            'predicted_label': predicted_labels, 
            'latency': latencies, 
            'model_name': print_model_names
        })
    else:
        df = pd.DataFrame({
            'actual_length': actual_lengths, 
            'predicted_label': predicted_labels, 
            'latency': latencies, 
            'turn_id': turn_ids, 
            'model_name': print_model_names
        })
    return df


def eval_all_models(model, testset, device, model_names=None, num_classes=3, task_type=0):
    """
    Evaluate model performance separately for each LLM model in the dataset.
    
    Args:
        model: Model to evaluate.
        testset: Dataset to evaluate on.
        device: Device to run evaluation on.
        model_names: List of model names for evaluation.
        num_classes: Number of classes for classification.
        task_type: Type of task (0=regression, 1=classification, etc.).
        
    Returns:
        List of metrics for each model.
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    import torch.nn as nn
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", average="macro")
    precision_metric = evaluate.load("precision", average="macro")
    recall_metric = evaluate.load("recall", average="macro")
    l1loss = nn.L1Loss()
    mseloss = nn.MSELoss()
    model.eval()
    
    # Getting tokenizer from model parameters
    model_config = model.config if hasattr(model, 'config') else None
    tokenizer_name = model_config.name_or_path if model_config else "bert-base-uncased"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metrics = []
    model_counts = [0 for _ in range(len(model_names))]

    for i in range(len(model_names)):
        data_subset = testset.filter(lambda example: example["model"][i] == 1)
        model_counts[i] = len(data_subset['model'])
        if len(data_subset['model']) == 0:
            metrics.append({})
            continue
        dataloader = DataLoader(data_subset, shuffle=True, batch_size=1, collate_fn=data_collator)
        predictions = []
        labels = []
        l1err = 0.0
        mse = 0.0

        for batch in dataloader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                model_name = batch['model'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
                label = batch['labels'].to(device)

                if task_type != 3 and task_type != 4:
                    prediction = torch.argmax(output, dim=-1)
                else:
                    prediction = torch.round(output).type(torch.LongTensor)
                    l1err += l1loss(output, label.type_as(output))
                    mse += mseloss(output, label.type_as(output))
                    for i in range(len(prediction)):
                        if prediction[i] >= num_classes:
                            prediction[i] = num_classes - 1
                        elif prediction[i] < 0:
                            prediction[i] = 0
                labels.extend(label)
                predictions.extend(prediction)
        reg_metric = {'L1 error': l1err.item() / len(dataloader), 'MSE': mse.item() / len(dataloader)}
        metric = accuracy_metric.compute(references=labels, predictions=predictions) | \
            f1_metric.compute(references=labels, predictions=predictions, average='macro') | \
            precision_metric.compute(references=labels, predictions=predictions, average='macro') | \
            recall_metric.compute(references=labels, predictions=predictions, average='macro') | \
            reg_metric
        metrics.append(metric)
    return metrics, model_counts


def plot_model_metrics(metrics, model_names, model_counts):
    """
    Create a plot of model performance metrics.
    
    Args:
        metrics: List of metric dictionaries for each model.
        model_names: List of model names.
        model_counts: Count of samples for each model.
        
    Returns:
        None, but saves plot and CSV files.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Filter to models with data
    remaining_model_names = []
    for i in range(len(model_names)):
        if model_counts[i] > 0:
            remaining_model_names.append(model_names[i])
            
    metrics_data = [[] for _ in range(4)]
    
    # Collect metrics for each model
    for i in range(len(model_names)):
        if model_counts[i] == 0:
            continue
        for j, (k, v) in enumerate(metrics[i].items()):
            if j >= 4:  # Only take the first 4 metrics (accuracy, f1, precision, recall)
                break
            metrics_data[j].append(v)

    # Create a DataFrame and save CSV
    df = pd.DataFrame({ 
        'Model Name': remaining_model_names,
        'Accuracy': metrics_data[0], 
        'F1 Score': metrics_data[1],
        'Precision': metrics_data[2],
        'Recall': metrics_data[3]
    }) 
    
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/cls_all_models_metrics.csv', index=False) 
    
    # Create and save plot
    ax = df.plot(x="Model Name", y=["Accuracy", "F1 Score", "Precision", "Recall"], 
                kind="bar", figsize=(20, 10)) 
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig("./results/cls_all_models.pdf") 