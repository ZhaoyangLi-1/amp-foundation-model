import logging
import esm

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from argparse import ArgumentParser
import json

from proteinchat.common.registry import registry
from proteinchat.models.blip2 import Blip2Base, disabled_train
from proteinchat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import re
import numpy as np
from .alphafold_utils import AlphaFoldPredictor
import os
from transformers import AutoTokenizer, EsmModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from accelerate import dispatch_model
avalaible_gpus = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")
device_af = torch.device(f'cuda:{avalaible_gpus[0]}')  # For AlphaFoldPredictor
device_pc = torch.device(f'cuda:{avalaible_gpus[1]}')  # For ProteinChat components

# device = torch.device('cuda')
# print(f'Using device: {device}')


@registry.register_model("proteinchat")
class ProteinChat(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "",
    }

    def __init__(
        self,
        freeze_protein_encoder=True,
        freeze_str_encoder=True,
        freeze_lp=False,
        freeze_llama=True,
        llama_model="",
        embedding_agg=1, 
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        alphafold_config_preset="model_1_ptm",
        alphafold_output_dir=None,  # Set appropriately
        alphafold_model_device="cuda:0",  # Assign device as needed
        alphafold_use_precomputed_alignments=None,  # Path if available
        alphafold_experiment_config_json=None,  # Path if available
        alphafold_long_sequence_inference=False,
        alphafold_use_deepspeed_evoformer_attention=False,
    ):
        super().__init__()
        self.model_device = device_pc
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.embedding_agg = embedding_agg
        
        print("Loading Protein Encoder")
        self.protein_encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.protein_encoder = self.protein_encoder.to(self.model_device)
        self.protein_tokenizer = alphabet.get_batch_converter()
        
        print("Loading Structure Encoder")
        self.str_encoder, self.str_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.str_encoder = self.str_encoder.to(self.model_device)
        self.str_encoder = self.str_encoder.encoder
        
        print("Loading AlphaFold")
        # self.alphafold = AlphaFoldPredictor()
        self.alphafold = AlphaFoldPredictor(
            config_preset=alphafold_config_preset,
            output_dir=alphafold_output_dir,
            model_device=alphafold_model_device,
            use_precomputed_alignments=alphafold_use_precomputed_alignments,
            experiment_config_json=alphafold_experiment_config_json,
            long_sequence_inference=alphafold_long_sequence_inference,
            use_deepspeed_evoformer_attention=alphafold_use_deepspeed_evoformer_attention
        )
        
        if freeze_protein_encoder:
            for name, param in self.protein_encoder.named_parameters():
                param.requires_grad = False
            self.protein_encoder = self.protein_encoder.eval()
            self.protein_encoder.train = disabled_train
            logging.info("freeze protein encoder")
        else:
            self.protein_encoder = self.protein_encoder.train()
            
        if freeze_str_encoder:
            for name, param in self.str_encoder.named_parameters():
                param.requires_grad = False
            self.str_encoder = self.str_encoder.eval()
            self.str_encoder.train = disabled_train
            logging.info("freeze str encoder")
        else:
            self.str_encoder = self.str_encoder.train()
        
        
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)

        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        if self.low_resource:
            print("Start Low Resource Mode")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': self.model_device},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            ).to(self.model_device)
        # Move LLAMA model to device
        # self.llama_model = self.llama_model.to(device)
        
        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            lora_target_modules: list[str] = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        self.glm_llama_proj = nn.Linear(
            1280, self.llama_model.config.hidden_size
        ).to(self.model_device)
        self.str_llama_proj = nn.Linear(
            512, self.llama_model.config.hidden_size
        ).to(self.model_device)
        if freeze_lp:
            for name, param in self.glm_llama_proj.named_parameters():
                param.requires_grad = False
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def reconstruct_protein(self, seqs):
        # Convert sequences into FASTA content format
        fasta_contents = [f">sequence_{i}\n{seq}" for i, seq in enumerate(seqs)]

        # Get structure predictions
        structures = self.alphafold.predict_structure(fasta_contents)

        coords, confidence, padding_mask = zip(*structures)
        coords = torch.tensor(np.concatenate(coords, axis=0))
        confidence = torch.tensor(np.concatenate(confidence, axis=0))
        padding_mask = torch.tensor(np.concatenate(padding_mask, axis=0))
        
        return coords, confidence, padding_mask


    def encode_str(self, str_tokens):
        coords, confidence, padding_mask = [x.to(self.model_device) for x in str_tokens]
        self.str_encoder = self.str_encoder.to(self.model_device)  

        if coords.dtype != self.str_encoder.embed_tokens.weight.dtype:
            coords = coords.to(self.str_encoder.embed_tokens.weight.dtype)
        if confidence.dtype != self.str_encoder.embed_tokens.weight.dtype:
            confidence = confidence.to(self.str_encoder.embed_tokens.weight.dtype)
        padding_mask = padding_mask.to(self.model_device)

        assert coords.device == confidence.device == padding_mask.device == self.str_encoder.embed_tokens.weight.device, \
            f"Mismatch in devices: coords={coords.device}, confidence={confidence.device}, padding_mask={padding_mask.device}, str_encoder={self.str_encoder.embed_tokens.weight.device}"

        str_tokens = self.str_encoder(
            coords=coords, confidence=confidence, encoder_padding_mask=~padding_mask  # Invert mask if necessary
        )
        str_tokens = str_tokens["encoder_out"][0]
        str_tokens = str_tokens.permute([1, 0, 2])

        if str_tokens.dtype != self.str_llama_proj.weight.dtype:
            str_tokens = str_tokens.to(self.str_llama_proj.weight.dtype)
            
        self.str_llama_proj = self.str_llama_proj.to(self.model_device)
        str_embeds = self.str_llama_proj(str_tokens.to(self.model_device))  # Project to LLAMA hidden size
        atts_llama = padding_mask.long().to(self.model_device)
        return str_embeds, atts_llama


    def encode_protein(self, seqs):
        batch_seqs = [('protein', seq) for seq in seqs]
        batch_labels, batch_strs, batch_tokens = self.protein_tokenizer(batch_seqs)
        batch_tokens = batch_tokens.to(self.model_device)
        
        self.protein_encoder = self.protein_encoder.to(self.model_device)
        protein_embeds = self.protein_encoder(
            batch_tokens, repr_layers=[33], return_contacts=True
        )["representations"][33].to(self.model_device)

        # Extract per-residue representations
        self.protein_encoder = self.protein_encoder.to(self.model_device)
        protein_embeds = self.protein_encoder(batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33].to(batch_tokens.device)

        # input llama is of shape [B, len, 5120]
        if protein_embeds.dtype != self.glm_llama_proj.weight.dtype:
            protein_embeds = protein_embeds.to(self.glm_llama_proj.weight.dtype)

        self.glm_llama_proj = self.glm_llama_proj.to(self.model_device)
        inputs_llama = self.glm_llama_proj(protein_embeds.squeeze(dim=2)).to(self.model_device)
        # atts_llama is of shape [B, len]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.model_device)
        return inputs_llama, atts_llama

    def prompt_list_wrap(self, img_embeds, atts_img, str_embeds, atts_str, prompt):
        if prompt:
            p_before_lst = []
            p_between_lst = []
            p_after_lst = []
            for p in prompt:
                # Split the prompt into parts based on '<proteinHere>' and '<structureHere>'
                splits = re.split(r'(<proteinHere>|<structureHere>)', p)
                if len(splits) == 5 and splits[1] == '<proteinHere>' and splits[3] == '<structureHere>':
                    p_before = splits[0]
                    p_between = splits[2]
                    p_after = splits[4]
                else:
                    raise ValueError("Prompt format is incorrect. Expected format with '<proteinHere>' and '<structureHere>' placeholders.")
                p_before_lst.append(p_before)
                p_between_lst.append(p_between)
                p_after_lst.append(p_after)
            # Tokenize and embed the parts
            p_before_tokens = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False
            ).to(self.model_device)
            p_between_tokens = self.llama_tokenizer(
                p_between_lst, return_tensors="pt", add_special_tokens=False
            ).to(self.model_device)
            p_after_tokens = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True
            ).to(self.model_device)
            
            # Get embeddings
            breakpoint()
            # self.llama_model = self.llama_model.to(self.model_device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
            p_between_embeds = self.llama_model.model.embed_tokens(p_between_tokens.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            
            # Now assemble the embeddings
            wrapped_embeds = torch.cat(
                [p_before_embeds, img_embeds, p_between_embeds, str_embeds, p_after_embeds], dim=1
            )
            # Adjust attention masks
            wrapped_atts = torch.cat(
                [
                    p_before_tokens.attention_mask,
                    atts_img,
                    p_between_tokens.attention_mask,
                    atts_str,
                    p_after_tokens.attention_mask,
                ],
                dim=1,
            )
            return wrapped_embeds, wrapped_atts
        else:
            # If no prompt, just concatenate img_embeds and str_embeds
            wrapped_embeds = torch.cat([img_embeds, str_embeds], dim=1)
            wrapped_atts = torch.cat([atts_img, atts_str], dim=1)
            return wrapped_embeds, wrapped_atts

    def forward(self, samples):
        seqs = samples["seq"]  # List of sequences
        str_tokens = self.reconstruct_protein(seqs)
        str_embeds, atts_str = self.encode_str(str_tokens)
        protein_embeds, atts_protein = self.encode_protein(seqs)

        # Use the revised prompt_list_wrap function
        img_embeds, atts_img = self.prompt_list_wrap(
            protein_embeds, atts_protein, str_embeds, atts_str, samples["prompt"]
        )

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(self.model_device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = torch.ones(
            [atts_img.shape[0], atts_img.shape[1] + 1], dtype=torch.long
        ).to(self.model_device).fill_(-100)  # Plus one for BOS
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        self.llama_model = self.llama_model.to(self.model_device)
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1
        )

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):

        llama_model = cfg.get("llama_model")

        freeze_protein_encoder = cfg.get("freeze_protein_encoder", False)
        freeze_str_encoder = cfg.get("freeze_str_encoder", False)
        freeze_lp = cfg.get("freeze_lp", False)
        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)
        alphafold_config_preset = cfg.get("alphafold_config_preset", "model_1_ptm")
        alphafold_output_dir = cfg.get("alphafold_output_dir", "/path/to/output_dir")
        alphafold_model_device = cfg.get("alphafold_model_device", "cuda:0")
        alphafold_use_precomputed_alignments = cfg.get("alphafold_use_precomputed_alignments", None)
        alphafold_experiment_config_json = cfg.get("alphafold_experiment_config_json", None)
        alphafold_long_sequence_inference = cfg.get("alphafold_long_sequence_inference", False)
        alphafold_use_deepspeed_evoformer_attention = cfg.get("alphafold_use_deepspeed_evoformer_attention", False)
        model = cls(
            freeze_protein_encoder=freeze_protein_encoder,
            freeze_str_encoder=freeze_str_encoder,
            freeze_lp=freeze_lp,
            freeze_llama=freeze_llama,
            llama_model=llama_model,
            embedding_agg = embedding_agg, 
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            alphafold_config_preset=alphafold_config_preset,
            alphafold_output_dir=alphafold_output_dir,
            alphafold_model_device=alphafold_model_device,
            alphafold_use_precomputed_alignments=alphafold_use_precomputed_alignments, 
            alphafold_experiment_config_json=alphafold_experiment_config_json, 
            alphafold_long_sequence_inference=alphafold_long_sequence_inference,
            alphafold_use_deepspeed_evoformer_attention=alphafold_use_deepspeed_evoformer_attention
        )

        stage1_ckpt = cfg.get("stage1_ckpt", "")  # load weights of encoder and LP
        if stage1_ckpt:
            print("Load GLM and LP Checkpoint: {}".format(stage1_ckpt))
            ckpt = torch.load(stage1_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        
        peft_ckpt = cfg.get("peft_ckpt", "")  # load weights of LoRA
        if peft_ckpt:
            print("Load LoRA Checkpoint: {}".format(peft_ckpt))
            ckpt = torch.load(peft_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
