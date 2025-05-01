import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import evaluate
from logger import Logger

def write_loss_to_file(training_loss_list, validation_loss_list, selected_data_size):
    """
    Write training and validation loss to a file.
    
    Args:
        training_loss_list: List of training losses.
        validation_loss_list: List of validation losses.
        selected_data_size: Size of dataset in number of examples.
    """
    cur_dir = os.path.dirname(__file__)
    train_dir = os.path.join(cur_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(os.path.join(train_dir, date_time + f'-size_{int(selected_data_size / 1000)}K.txt'), 'w') as f:
        f.write('Training loss:\n')
        for loss in training_loss_list:
            f.write(str(loss) + '\t')
        f.write('\nValidation loss:\n')
        for loss in validation_loss_list:
            f.write(str(loss) + '\t')
        f.write('\n')


def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device, 
          flag_bert_tuning=True, flag_vicuna_data_only=False, task_type=0, flag_write_results=False, 
          selected_data_size=1000, logger=None):
    """
    Train the model.
    
    Args:
        model: Model to train.
        criterion: Loss function.
        optimizer: Optimizer.
        train_dataloader: DataLoader for training data.
        validation_dataloader: DataLoader for validation data.
        num_epochs: Number of epochs to train for.
        device: Device to train on.
        flag_bert_tuning: Whether to tune BERT weights.
        flag_vicuna_data_only: Whether using only vicuna data.
        task_type: Type of task (0=regression, 1=binary, etc.).
        flag_write_results: Whether to write results to TensorBoard.
        selected_data_size: Size of dataset used.
        logger: Logger for tracking experiments.
        
    Returns:
        None
    """
    num_training_steps = num_epochs * len(train_dataloader)
    # Using a learning rate with a linear decay
    lr_scheduler = transformers.get_scheduler(
        'linear',
        # 'constant',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    training_loss_list = []
    validation_loss_list = []
    if flag_write_results:
        writer = SummaryWriter()

    # Log hyperparameters if logger is available
    if logger:
        hyperparams = {
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_dataloader.batch_size if hasattr(train_dataloader, 'batch_size') else None,
            "epochs": num_epochs,
            "bert_tuning": flag_bert_tuning,
            "vicuna_data_only": flag_vicuna_data_only,
            "task_type": task_type,
            "dataset_size": selected_data_size,
            "model_type": model.__class__.__name__,
            "criterion": criterion.__class__.__name__
        }
        logger.log_hyperparams(hyperparams)

    for epoch in tqdm(range(num_epochs)):
        training_loss = 0
        model.train()
        # Fix the BERT weights after 3 training epochs
        if flag_bert_tuning and epoch == 3:
            for param in model.bert.parameters():
                param.requires_grad = False
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
            if logger:
                logger.log_metrics({"learning_rate": 1e-4}, step=epoch)

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if flag_vicuna_data_only:
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_name = batch['model'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
            if task_type == 0:
                labels = batch['num_tokens'].to(device)
            else:
                labels = batch['labels'].to(device)
            if task_type == 0 or task_type == 3 or task_type == 4:
                loss = criterion(output, labels.float())
            else:
                loss = criterion(output, labels)
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            training_loss += loss.item()

        epoch_train_loss = training_loss / len(train_dataloader)
        
        if flag_write_results:
            writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        
        print(f"Training loss for epoch {epoch}: {epoch_train_loss}")
        training_loss_list.append(epoch_train_loss)
        
        if epoch % 1 == 0:
            if task_type == 0:
                validation_metrics = eval_regression(model, validation_dataloader, device, flag_vicuna_data_only)
            elif task_type == 3 or task_type == 4:
                validation_metrics = eval_regression(model, validation_dataloader, device, flag_vicuna_data_only)
                validation_metrics = validation_metrics | eval_classification(model, validation_dataloader, device, flag_vicuna_data_only)
            else:
                validation_metrics = eval_classification(model, validation_dataloader, device, flag_vicuna_data_only)
            
            # Log the metrics if logger is available
            if logger:
                metrics_to_log = {
                    "train/loss": epoch_train_loss,
                    **{f"val/{k}": v for k, v in validation_metrics.items()}
                }
                logger.log_metrics(metrics_to_log, step=epoch)
            
            print(f'Validation loss after epoch {epoch}: ')
            for k, v in validation_metrics.items():
                print(f'{k}: {v:.4f}', end='\t')
            print(' ')
            
    if flag_write_results:
        writer.flush()
        writer.close()
        write_loss_to_file(training_loss_list, validation_loss_list, selected_data_size)


def eval_classification(model, dataloader, device, flag_vicuna_data_only=False):
    """
    Evaluate classification model.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader for evaluation data.
        device: Device to evaluate on.
        flag_vicuna_data_only: Whether using only vicuna data.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", average="macro")
    precision_metric = evaluate.load("precision", average="macro")
    recall_metric = evaluate.load("recall", average="macro")
    model.eval()
    labels = []
    predictions = []
    for batch in dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if flag_vicuna_data_only:
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_name = batch['model'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
            label = batch['labels'].to(device)

            # Determine if this is a regression or classification task
            if not hasattr(model, 'fc2') or model.fc2.out_features > 1:
                prediction = torch.argmax(output, dim=-1)
            else:
                # This is a regression task being used for classification
                prediction = torch.round(output).type(torch.LongTensor)
                num_classes = getattr(model, 'fc2').out_features if hasattr(model, 'fc2') else 3
                for i in range(len(prediction)):
                    if prediction[i] >= num_classes:
                        prediction[i] = num_classes - 1
                    elif prediction[i] < 0:
                        prediction[i] = 0
            labels.extend(label)
            predictions.extend(prediction)
    metric = accuracy_metric.compute(references=labels, predictions=predictions) | \
        f1_metric.compute(references=labels, predictions=predictions, average='macro') | \
        precision_metric.compute(references=labels, predictions=predictions, average='macro') | \
        recall_metric.compute(references=labels, predictions=predictions, average='macro')
    return metric


def eval_regression(model, dataloader, device, flag_vicuna_data_only=False):
    """
    Evaluate regression model.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader for evaluation data.
        device: Device to evaluate on.
        flag_vicuna_data_only: Whether using only vicuna data.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    l1loss = nn.L1Loss()
    mseloss = nn.MSELoss()
    model.eval()

    l1err = 0
    mse = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if flag_vicuna_data_only:
                prediction = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_name = batch['model'].to(device)
                prediction = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
            if hasattr(batch, 'num_tokens'):
                labels = batch['num_tokens'].to(device)
            else:
                labels = batch['labels'].to(device)
            l1err += l1loss(prediction, labels.type_as(prediction))
            mse += mseloss(prediction, labels.type_as(prediction))

    metric = {'L1 error': l1err.item() / len(dataloader), 'MSE': mse.item() / len(dataloader)}
    return metric 