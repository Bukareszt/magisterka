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
        self.vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_name, use_fast=False, trust_remote_code=True)

        # ⚡ Vicuna (na GPU + float16 jeśli CUDA)
        self.vicuna = AutoModelForCausalLM.from_pretrained(
            vicuna_name,
            trust_remote_code=True,
            device_map="auto"
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

        # Tokenizacja promptów
        inputs = self.vicuna_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.vicuna.device)
        attention_mask = inputs["attention_mask"].to(self.vicuna.device)

        # Początkowy forward przez prompt (bez generowania)
        with torch.no_grad():
            outputs = self.vicuna(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            generated_ids = input_ids
            collected_hidden_states = [outputs.hidden_states[-1]]  # initial hidden states

            last_token = input_ids[:, -1:]

            for _ in range(n_tokens):
                # Generujemy następny token
                next_outputs = self.vicuna(
                    input_ids=last_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                # Aktualizacja pastów
                past_key_values = next_outputs.past_key_values

                # Zapisujemy ostatni hidden state
                collected_hidden_states.append(next_outputs.hidden_states[-1])

                # Zakładamy, że model poda logits → wybieramy token do symulacji
                next_token = torch.argmax(next_outputs.logits[:, -1, :], dim=-1, keepdim=True)
                last_token = next_token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Po `n_tokens` generacji: zbieramy wszystkie hidden states
        all_hidden = torch.cat(collected_hidden_states[1:], dim=1)  # Pomiń pierwszy prompt
        # [B, n_tokens, D_vicuna] – symulowane tokeny

        # Adapter: Vicuna → BERT
        bert_input = self.adapter(all_hidden.to(self.device))  # [B, n_tokens, D_bert]
        attention_mask = torch.ones(bert_input.shape[:2], dtype=torch.long, device=self.device)

        # BERT forward
        bert_output = self.bert(inputs_embeds=bert_input, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]  # Można zmienić na mean()

        # Regressor
        return self.regressor(cls_token).squeeze(-1) * self.max_output_length

    def predict(self, prompts, n_tokens=1):
        self.eval()
        with torch.no_grad():
            preds = self.forward(prompts, n_tokens=n_tokens)
        return preds.item() if isinstance(prompts, str) else preds.tolist()
