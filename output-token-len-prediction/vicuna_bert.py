import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


class VicunaToBertRegressor(nn.Module):
    def __init__(self, vicuna_name="lmsys/vicuna-13b-v1.3", bert_name="bert-base-uncased", max_output_length=512):
        super().__init__()

        # 🔍 Wybór najlepszego dostępnego urządzenia
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        print(f"[INFO] Using device: {self.device}")

        # Store max_output_length for scaling
        self.max_output_length = max_output_length

        # Tokenizer Vicuny
        self.vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_name, use_fast=False)

        # ⚡ Vicuna (na GPU + float16 jeśli CUDA)
        self.vicuna = AutoModelForCausalLM.from_pretrained(
            vicuna_name,
            trust_remote_code=True,
            device_map="auto"  # korzystamy z automatycznego mapowania na GPU
        )
        for param in self.vicuna.parameters():
            param.requires_grad = False

        self.vicuna_output_dim = self.vicuna.config.hidden_size

        # 🔁 BERT (na tym samym urządzeniu co reszta)
        self.bert = AutoModel.from_pretrained(bert_name)
        self.bert.to(self.device)
        self.bert_input_dim = self.bert.config.hidden_size

        # 🔄 Adapter Vicuna → BERT
        self.adapter = nn.Linear(self.vicuna_output_dim, self.bert_input_dim)
        self.adapter.to(self.device)

        # 🔚 Regressor - modified to output values between 0 and max_output_length
        self.regressor = nn.Sequential(
            nn.Linear(self.bert_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Will constrain output to [0, 1]
        )
        self.regressor.to(self.device)

    def forward(self, prompts, n_tokens=1):
        if isinstance(prompts, str):
            prompts = [prompts]

        # 🔠 Tokenizacja
        inputs = self.vicuna_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        for k in inputs:
            inputs[k] = inputs[k].to(self.vicuna.device)  # Vicuna może być na wielu GPU

        # 🧠 Generowanie i hidden states
        with torch.no_grad():
            vicuna_outputs = self.vicuna(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        # Hidden states z ostatniej warstwy (tylko inputy)
        last_hidden_states = vicuna_outputs.hidden_states[-1]  # shape: [B, T, D_vicuna]
        gen_hidden = last_hidden_states[0][:, -n_tokens:, :]  # [B, n_tokens, D_vicuna]

        # ⚙️ Adapter → BERT
        gen_hidden = gen_hidden.to(self.device)
        bert_input = self.adapter(gen_hidden)  # [B, n_tokens, D_bert]
        attention_mask = torch.ones(bert_input.shape[:2], dtype=torch.long, device=self.device)

        # 🧠 BERT + regresja
        bert_output = self.bert(inputs_embeds=bert_input, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]

        # Scale the sigmoid output to range [0, max_output_length]
        return self.regressor(cls_token).squeeze(-1) * self.max_output_length

    def predict(self, prompts, n_tokens=1):
        self.eval()
        with torch.no_grad():
            preds = self.forward(prompts, n_tokens=n_tokens)
        return preds.item() if isinstance(prompts, str) else preds.tolist()
