import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import OrderedDict
from transformers.models.llama import LlamaForCausalLM, LlamaModel
from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from accelerate import hooks
from torch.nn.modules import module

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


class Llama4Code(nn.Module):
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
        freeze_emb_table=True,
        device_map="auto",
    ):
        super(Llama4Code, self).__init__()

        self.input_dim, self.output_dim = input_dim, output_dim
        self.lang = language

        print(f"Initializing language decoder ...")

        self.llama_model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            use_safetensors=False,
        )
        print(self.llama_model.device)
        if load_in_8bit:
            self.llama_model = prepare_model_for_int8_training(self.llama_model)
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
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            print("Lora used")
        self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            model_path, add_eos_token=True
        )
        self.llama_tokenizer.pad_token = (
            "<unk>"  # This is the <unk> token, instead of eos token(id=1, <\s>)
        )
        self.llama_tokenizer.padding_side = "right"
        print("Language decoder initialized.")

        # if freeze_emb_table:
        #     for name, param in self.item_embedding.named_parameters():
        #         param.requires_grad = False
        #         self.item_embedding = self.item_embedding.eval()
        #         print("Freeze item embedding table")

        self.embedding_proj = nn.Linear(
            self.input_dim, self.llama_model.config.hidden_size
        )

    def forward(self, *args, **kwargs):
        input_ids, labels, attention_mask, encoded_emb = (
            kwargs["input_ids"],
            kwargs["labels"],
            kwargs["attention_mask"],
            kwargs["encoded_emb"],  # [bs, hist_lenth]
        )

        bs, seq_lenth = input_ids.shape[0], input_ids.shape[1]
        unk_token_id = self.llama_tokenizer.unk_token_id
        replaced_idx = torch.nonzero(
            input_ids == unk_token_id
        )  # shape [Num of index, bs]
        remain_idx = torch.nonzero(input_ids != unk_token_id)
        prompt_embeds = self.llama_model.base_model.model.model.embed_tokens(
            input_ids[
                remain_idx[:, 0],
                remain_idx[:, 1],
            ]
        )  # [bs, seq_lenth, embedding_size]
        x_emb = torch.zeros([bs, seq_lenth, 5120], dtype=torch.bfloat16).to(
            prompt_embeds.device
        )
        item_embedding = self.embedding_proj(encoded_emb).view(-1, self.output_dim)

        x_emb[replaced_idx[:, 0], replaced_idx[:, 1], :] = item_embedding.to(
            torch.bfloat16
        ).to(prompt_embeds.device)
        x_emb[remain_idx[:, 0], remain_idx[:, 1], :] = prompt_embeds.to(torch.bfloat16)
        assert (
            attention_mask.shape[0] == x_emb.shape[0]
            and attention_mask.shape[1] == x_emb.shape[1]
        )
        return self.llama_model.forward(
            inputs_embeds=x_emb.to(torch.float),
            attention_mask=attention_mask.to(torch.float),
            return_dict=True,
            labels=labels,
        )

    def get_input_emb(self, input_ids, attention_mask, encoded_emb):
        bs, seq_lenth = input_ids.shape[0], input_ids.shape[1]
        unk_token_id = self.llama_tokenizer.unk_token_id
        replaced_idx = torch.nonzero(
            input_ids == unk_token_id
        )  # shape [Num of index, bs]
        remain_idx = torch.nonzero(input_ids != unk_token_id)
        prompt_embeds = self.llama_model.base_model.model.model.embed_tokens(
            input_ids[
                remain_idx[:, 0],
                remain_idx[:, 1],
            ]
        )  # [bs, seq_lenth, embedding_size]
        x_emb = torch.zeros([bs, seq_lenth, 5120], dtype=torch.bfloat16).to(
            prompt_embeds.device
        )
        item_embedding = self.embedding_proj(encoded_emb).view(-1, self.output_dim)

        x_emb[replaced_idx[:, 0], replaced_idx[:, 1], :] = item_embedding.to(
            torch.bfloat16
        ).to(prompt_embeds.device)
        x_emb[remain_idx[:, 0], remain_idx[:, 1], :] = prompt_embeds.to(torch.bfloat16)
        assert (
            attention_mask.shape[0] == x_emb.shape[0]
            and attention_mask.shape[1] == x_emb.shape[1]
        )
        return x_emb.to(torch.float)
