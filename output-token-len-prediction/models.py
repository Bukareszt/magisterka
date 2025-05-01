import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertClassificationModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim, num_classes, flag_bert_tuning=False, flag_vicuna_data_only=False, num_models=25):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        self.flag_bert_tuning = flag_bert_tuning
        self.flag_vicuna_data_only = flag_vicuna_data_only
        self.num_models = num_models
        
        # Fix the weights of the pretrained model
        if not self.flag_bert_tuning:
            for param in self.bert.parameters():
                param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        if self.flag_vicuna_data_only:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim + self.num_models, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        if self.flag_vicuna_data_only:
            output = self.relu(self.fc1(output))
        else:
            output = self.relu(self.fc1(torch.cat((output, model_name), dim=-1)))
        output = self.logsoftmax(self.fc2(output))
        return output
    

class BertRegressionModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim, flag_bert_tuning=False, flag_vicuna_data_only=False, num_models=25):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        self.flag_bert_tuning = flag_bert_tuning
        self.flag_vicuna_data_only = flag_vicuna_data_only
        self.num_models = num_models
        
        # Fix the weights of the pretrained model
        if not self.flag_bert_tuning:
            for param in self.bert.parameters():
                param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        if self.flag_vicuna_data_only:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim + self.num_models, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        if self.flag_vicuna_data_only:
            output = self.relu(self.fc1(output))
        else:
            output = self.relu(self.fc1(torch.cat((output, model_name), dim=-1)))
        output = self.fc2(output).squeeze(-1)
        return output 