import os
import re

def get_output_file_name(flag_first_round_only=True, flag_vicuna_data_only=False, 
                          flag_bert_tuning=False, flag_tiny_bert=False, 
                          task_type=0, flag_l1_loss=False, selected_data_size=1000, 
                          model_name=None, flag_head_tail=False, add_response_tokens=0):
    """
    Generate output filename based on configuration parameters.
    
    Args:
        flag_first_round_only: Whether using only first round data.
        flag_vicuna_data_only: Whether using only vicuna data.
        flag_bert_tuning: Whether BERT is being tuned.
        flag_tiny_bert: Whether using tiny BERT.
        task_type: Type of task (0=regression, 1=classification, etc.).
        flag_l1_loss: Whether using L1 loss.
        selected_data_size: Size of dataset used.
        model_name: Name of LLM model being used.
        flag_head_tail: Whether using head and tail.
        add_response_tokens: Number of response tokens added.
        
    Returns:
        String representing the output filename.
    """
    output_filename = 'predictions_'
    if not flag_first_round_only:
        if flag_head_tail:
            output_filename += 'multiround_headtail_'
        else:
            output_filename += 'multiround_tail_'
    if not flag_vicuna_data_only:
        output_filename += 'all_models_'
    else:
        if model_name:
            output_filename += model_name.lower() + '_'
        else:
            output_filename += 'vicuna-13b_'
    if flag_bert_tuning:
        output_filename += 'warmup_'
    if flag_tiny_bert:
        output_filename += 'berttiny_'
    if task_type == 0:
        output_filename += 'reg_'
        output_filename += 'l1_' if flag_l1_loss else 'mse_'
    elif task_type == 1:
        output_filename += 'cls_'
    elif task_type == 2:
        output_filename += 'multi_cls_'
    elif task_type == 3:
        output_filename += 'ordinal_multi_cls_'
        output_filename += 'l1_' if flag_l1_loss else 'mse_'
    elif task_type == 4:
        output_filename += 'ordinal_cls_'
        output_filename += 'l1_' if flag_l1_loss else 'mse_'
        
    output_filename += f'preview{add_response_tokens}_'
    output_filename += f'{int(selected_data_size / 1000)}K.csv'
    return output_filename


def get_dataset_path(flag_first_round_only=True, flag_vicuna_data_only=False, 
                     task_type=0, selected_data_size=1000, model_name=None,
                     flag_head_tail=False, customized_path=None, add_response_tokens=0):
    """
    Generate path to dataset based on configuration parameters.
    
    Args:
        flag_first_round_only: Whether using only first round data.
        flag_vicuna_data_only: Whether using only vicuna data.
        task_type: Type of task (0=regression, 1=classification, etc.).
        selected_data_size: Size of dataset used.
        model_name: Name of LLM model being used.
        flag_head_tail: Whether using head and tail.
        customized_path: Custom dataset path to use (overrides other parameters).
        add_response_tokens: Number of response tokens added.
        
    Returns:
        String representing the dataset path.
    """
    if customized_path:
        return customized_path
        
    model_name_part = (model_name.lower() + '_') if flag_vicuna_data_only and model_name else ''
    
    # Add preview token part right after model name if applicable
    if add_response_tokens > 0:
        model_name_part += f'preview{add_response_tokens}_'
    
    if flag_first_round_only:
        round_part = 'first_round_data_'
    elif flag_head_tail:
        round_part = 'headtail_'
    else:
        round_part = 'tail_'
        
    if task_type == 0:
        dataset_path = 'data/lmsys_' + round_part + model_name_part + f'{int(selected_data_size / 1000)}K'
    elif task_type == 1 or task_type == 4:
        dataset_path = 'data/lmsys_' + round_part + model_name_part + f'cls_{int(selected_data_size / 1000)}K'
    elif task_type == 2 or task_type == 3:
        # multi_cls or ordinal_cls:
        dataset_path = 'data/lmsys_' + round_part + model_name_part + f'multi_cls_{int(selected_data_size / 1000)}K'
    
    return dataset_path 

def extract_preview_tokens_from_dataset_path(dataset_path):
    """
    Extract the number of preview tokens from a dataset path.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        int: Number of preview tokens or 0 if not found
    """
    # Look for preview pattern in the dataset path
    preview_pattern = re.compile(r'preview(\d+)')
    match = preview_pattern.search(dataset_path)
    
    if match:
        return int(match.group(1))
    return 0 