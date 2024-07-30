import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import OrderedDict
from transformers.models.starcoder2 import Starcoder2ForCausalLM, Starcoder2Model
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.starcoder2.modeling_starcoder2 import CausalLMOutputWithPast
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


class starcoder4Code(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        load_in_8bit,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target_modules,
        model_path,
        language="c++",
        device_map="auto",
    ):
        super(starcoder4Code, self).__init__()

        self.input_dim, self.output_dim = input_dim, output_dim
        self.lang = language
        self.hidden_dim = output_dim

        print(f"Initializing language decoder ...")

        self.starcoder_model = Starcoder2ForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
        )
        print(self.starcoder_model.device)
        if load_in_8bit:
            self.starcoder_model = prepare_model_for_int8_training(self.starcoder_model)
        if use_lora:
            # add the lora module
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.starcoder_model = get_peft_model(self.starcoder_model, peft_config)
            print("Lora used")
        self.starcoder_model.print_trainable_parameters()
        self.starcoder_model.config.use_cache = False

        self.starcoder_tokenizer = GPT2TokenizerFast.from_pretrained(
            model_path, add_eos_token=True
        )

        self.starcoder_tokenizer.pad_token = self.starcoder_tokenizer.eos_token
        self.starcoder_tokenizer.padding_side = "right"
        print("Language decoder initialized.")

        self.embedding_proj = nn.Linear(
            self.input_dim, self.starcoder_model.config.hidden_size
        )

    def forward(self, *args, **kwargs):
        input_ids, labels, attention_mask, encoded_emb = (
            kwargs["input_ids"],
            kwargs["labels"],
            kwargs["attention_mask"],
            kwargs["encoded_emb"],  # [bs, hist_lenth]
        )

        bs, seq_lenth = input_ids.shape[0], input_ids.shape[1]
        unk_token_id = self.starcoder_tokenizer.unk_token_id
        replaced_idx = torch.nonzero(
            input_ids == unk_token_id
        )  # shape [Num of index, bs]
        remain_idx = torch.nonzero(input_ids != unk_token_id)
        prompt_embeds = self.starcoder_model.base_model.model.model.embed_tokens(
            input_ids[
                remain_idx[:, 0],
                remain_idx[:, 1],
            ]
        )  # [bs, seq_lenth, embedding_size]
        x_emb = torch.zeros([bs, seq_lenth, self.hidden_dim]).to(prompt_embeds.device)
        item_embedding = self.embedding_proj(encoded_emb).view(-1, self.output_dim)

        x_emb[replaced_idx[:, 0], replaced_idx[:, 1], :] = item_embedding
        x_emb[remain_idx[:, 0], remain_idx[:, 1], :] = prompt_embeds
        assert (
            attention_mask.shape[0] == x_emb.shape[0]
            and attention_mask.shape[1] == x_emb.shape[1]
        )
        return self.starcoder_model.forward(
            inputs_embeds=x_emb,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )

    def get_input_emb(self, input_ids, attention_mask, encoded_emb):
        bs, seq_lenth = input_ids.shape[0], input_ids.shape[1]
        unk_token_id = self.starcoder_tokenizer.unk_token_id
        replaced_idx = torch.nonzero(
            input_ids == unk_token_id
        )  # shape [Num of index, bs]
        remain_idx = torch.nonzero(input_ids != unk_token_id)
        prompt_embeds = self.starcoder_model.base_model.model.model.embed_tokens(
            input_ids[
                remain_idx[:, 0],
                remain_idx[:, 1],
            ]
        )  # [bs, seq_lenth, embedding_size]
        x_emb = torch.zeros([bs, seq_lenth, self.hidden_dim]).to(prompt_embeds.device)
        item_embedding = self.embedding_proj(encoded_emb).view(-1, self.output_dim)

        x_emb[replaced_idx[:, 0], replaced_idx[:, 1], :] = item_embedding.to(
            prompt_embeds.device
        )
        x_emb[remain_idx[:, 0], remain_idx[:, 1], :] = prompt_embeds.to(
            prompt_embeds.device
        )
        assert (
            attention_mask.shape[0] == x_emb.shape[0]
            and attention_mask.shape[1] == x_emb.shape[1]
        )
        return x_emb.to(torch.float)
