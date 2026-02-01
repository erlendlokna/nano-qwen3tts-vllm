import torch
from nano_qwen3tts_vllm.engine.llm_engine.base import LLMEngine
from nano_qwen3tts_vllm.engine.sequence import Sequence, SequenceStatus
from nano_qwen3tts_vllm.engine.scheduler import Scheduler
from nano_qwen3tts_vllm.sampling_params import SamplingParams
from nano_qwen3tts_vllm.engine.model_runner.talker_mode_runner import TalkerModeModelRunner


from nano_qwen3tts_vllm.config import Config

class TalkerScheduler(Scheduler):
    def __init__(self, config: Config):
        super().__init__(config)
        self.request_id_to_seq: dict[str, Sequence] = {}

    def clear_request(self, request_id: str):
        if request_id in self.request_id_to_seq:
            seq = self.request_id_to_seq.pop(request_id)
            self.block_manager.deallocate(seq)
            if seq in self.running:
                self.running.remove(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], hidden_states: list[torch.Tensor]):
        idx = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id, hidden_states[idx])
            idx += 1
            if seq.request_id is not None:
                finish = not seq.ignore_eos and token_id == self.eos
            else:
                finish = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens
            if finish:
                seq.status = SequenceStatus.FINISHED
                if seq.request_id is not None:
                    self.request_id_to_seq.pop(seq.request_id, None)
                self.block_manager.deallocate(seq)
                self.running.remove(seq)



class TalkerLLMEngine(LLMEngine):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.model_runner = TalkerModeModelRunner(self.config, 0, self.events)
        self.scheduler = TalkerScheduler(self.config)

    def add_request(
        self,
        inputs_embeds: list[torch.Tensor],
        sampling_params: SamplingParams | list[SamplingParams],
        request_id: str | None = None,
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(inputs_embeds)
        for inp_embeds, sp in zip(inputs_embeds, sampling_params):
            if request_id is not None and request_id in self.scheduler.request_id_to_seq:
                seq = self.scheduler.request_id_to_seq[request_id]
                seq.decode_input_embeds = inp_embeds
                return
            seq = Sequence([], input_embeds=inp_embeds, sampling_params=sp, request_id=request_id)
            if request_id is not None:
                self.scheduler.request_id_to_seq[request_id] = seq
            self.scheduler.add(seq)

    def clear_request(self, request_id: str):
        self.scheduler.clear_request(request_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids, hidden_states = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, hidden_states)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.last_hidden_state) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def step_with_outputs(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids, hidden_states = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, hidden_states)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.last_hidden_state) for seq in seqs if seq.is_finished]
        outputs_all = [(seq.seq_id, seq.completion_token_ids, seq.last_hidden_state, seq.is_finished) for seq in seqs]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens, outputs_all
            