import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class VisualProjector(nn.Module):
    def __init__(self, clip_dim, llm_dim, n_visual_tokens=1):
        super().__init__()
        self.n = n_visual_tokens
        self.proj = nn.Linear(clip_dim, llm_dim * n_visual_tokens)

    def forward(self, clip_feat):
        x = self.proj(clip_feat)              # [B, llm_dim * n]
        x = x.view(x.size(0), self.n, -1)    # [B, n, llm_dim]
        return x

class MultimodalPolicy(nn.Module):
    def __init__(self, model_name, clip_dim, n_visual_tokens=1, deci_hidden=256, action_size=5):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        llm_dim = self.llm.config.hidden_size

        self.projector = VisualProjector(clip_dim, llm_dim, n_visual_tokens=n_visual_tokens)

        for p in self.llm.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj","v_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)

        self.deci = nn.Sequential(
            nn.Linear(llm_dim, deci_hidden),
            nn.ReLU(),
            nn.Linear(deci_hidden, action_size)
        )

    def forward(self, input_ids=None, clip_feat=None, attention_mask=None):
        batch = clip_feat.size(0) # type: ignore
        visual_embeds = self.projector(clip_feat)  # [B, n_vis, llm_dim]

        if input_ids is None:
            batch_size = clip_feat.size(0) # type: ignore
            dummy_texts = [""] * batch_size
            tokens = self.tokenizer(dummy_texts, return_tensors="pt", padding=True)
            input_ids = tokens.input_ids.to(clip_feat.device) # type: ignore

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, D] # type: ignore
        inputs_embeds = torch.cat([visual_embeds, input_embeds], dim=1)  # [B, n_vis+L, D]

        outputs = self.llm(
                inputs_embeds=inputs_embeds,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True
            )

        last_hidden = outputs.hidden_states[-1] 
        pooled = last_hidden[:, 0, :]  # [B, D]
        logits = self.deci(pooled)
        return logits
