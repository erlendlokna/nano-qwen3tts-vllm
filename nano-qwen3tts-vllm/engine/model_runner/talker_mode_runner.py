import os
import json

import torch
from typing import Optional
import time
from safetensors.torch import load_file

from nano_qwen3tts_vllm.engine.model_runner.base import ModelRunner
from nano_qwen3tts_vllm.config import Qwen3TTSConfig
from nano_qwen3tts_vllm.models.qwen3_tts_talker import Qwen3TTSTalkerForCausalLM
from nano_qwen3tts_vllm.engine.sequence import Sequence, SequenceStatus
from nano_qwen3tts_vllm.sampling_params import SamplingParams

from nano_qwen3tts_vllm.utils.context import set_context, get_context, reset_context
from nano_qwen3tts_vllm.config import Config
from multiprocessing.synchronize import Event


class TalkerModeModelRunner(ModelRunner):
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        super().__init__(config, rank, event)
        self.model = self.load_model(config)
        self.post_init(rank)

    def load_model(self, config: Config):
        with open(os.path.join(config.model, "config.json"), "r") as f:
            model_config = json.load(f)
            model_config = Qwen3TTSConfig(**model_config)
        
        self.full_config = model_config
            
        model = Qwen3TTSTalkerForCausalLM(model_config.talker_config)
        
        self.model_config = model_config.talker_config
        
        state_dict = load_file(
            os.path.join(config.model, "model.safetensors")
        )
        model.load_state_dict(state_dict)   
        return model

    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, input_embeds: Optional[torch.Tensor] = None):
        model_input = input_embeds if input_embeds is not None else input_ids
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(model_input, positions)
        else:
            bs = input_embeds.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_embeds"][:bs] = model_input
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            
            hidden_states = graph_vars["outputs"][:bs]
            
        logits = self.model.compute_logits(hidden_states)
        
        if is_prefill:
            context = get_context()
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()

        return logits, hidden_states

    def prepare_decode_talker(self, seqs: list[Sequence]):
        positions = []
        slot_mapping = []
        context_lens = []
        input_embeds_list = []
        for seq in seqs:
            emb = seq.decode_input_embeds
            if emb is None:
                raise ValueError(f"Sequence {seq.seq_id} has no decode_input_embeds set")
            input_embeds_list.append(emb)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_embeds = torch.cat([e.reshape(-1, e.shape[-1]) if e.dim() > 1 else e.unsqueeze(0) for e in input_embeds_list], dim=0).to(dtype=torch.bfloat16)
        if input_embeds.device.type != "cuda":
            input_embeds = input_embeds.pin_memory().cuda(non_blocking=True)
        else:
            input_embeds = input_embeds.cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        input_ids = torch.zeros(len(seqs), dtype=torch.int64, device="cuda")
        return input_ids, input_embeds, positions

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_embeds = None
        if is_prefill:
            input_ids, input_embeds, positions = self.prepare_prefill(seqs)
        else:
            input_ids, input_embeds, positions = self.prepare_decode_talker(seqs)
            
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits, hidden_states = self.run_model(input_ids, positions, is_prefill, input_embeds)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids, hidden_states
    
    
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = self.model_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        input_embeds = torch.zeros(max_bs, hf_config.hidden_size)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_embeds=input_embeds,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

