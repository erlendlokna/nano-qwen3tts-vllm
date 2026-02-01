import json
import os
import sys
import uuid
import torch
import torch.distributed as dist
import numpy as np
import soundfile as sf
import time
from tqdm import tqdm
from nano_qwen3tts_vllm.utils.context import set_context
from nano_qwen3tts_vllm.utils.prompt import prepare_custom_voice_prompt
from nano_qwen3tts_vllm.processor import Qwen3TTSProcessor
from nano_qwen3tts_vllm.utils.generation import prepare_inputs
from nano_qwen3tts_vllm.llm import TalkerLLM, PredictorLLM
from nano_qwen3tts_vllm.sampling_params import SamplingParams
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)


torch.manual_seed(42)
# Use Qwen3-TTS processor when available so tokenization matches Qwen3TTSModel.generate_custom_voice exactly
def _get_processor(model_path: str):
    try:
        from qwen_tts.core.models import Qwen3TTSProcessor as Qwen3TTSProcessorHF
        return Qwen3TTSProcessorHF.from_pretrained(model_path, fix_mistral_regex=True)
    except ImportError:
        return Qwen3TTSProcessor.from_pretrained(model_path, fix_mistral_regex=True)


class Qwen3TTSInterface:
    def __init__(self, model_path: str, enforce_eager: bool = False, tensor_parallel_size: int = 1):
        self.model_path = model_path
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size
        self.talker_llm = TalkerLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.3)
        self.predictor_llm = PredictorLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size)
        self.processor = _get_processor(model_path)
        self.model_config = self.talker_llm.model_runner.full_config
        
        self.text_embedding = self.talker_llm.model_runner.model.get_text_embeddings()
        self.input_embedding = self.talker_llm.model_runner.model.get_input_embeddings()
        self.text_projection = self.talker_llm.model_runner.model.text_projection
        
        self.predictor_input_embeddings = self.predictor_llm.model_runner.model.model.codec_embedding
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def generate_custom_voice(self, text: str, language: str = "English", speaker: str = "Vivian"):
        # Align with Qwen3TTSModel.generate_custom_voice: same _build_assistant_text, _tokenize_texts, device
        input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
            text=text,
            language=language,
            speaker=speaker,
            processor=self.processor,
            device=self.device,
        )
        
        # Qwen3-TTS generate_custom_voice uses non_streaming_mode=True; must match so talker gets full prompt (18 tokens)
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = prepare_inputs(
            config=self.model_config,
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            speakers=speakers,
            languages=languages,
            non_streaming_mode=True,
            text_embedding=self.text_embedding,
            input_embedding=self.input_embedding,
            text_projection=self.text_projection,
            device=self.device,
        )
        
        
        # talker_input_embeds from prepare_inputs is (batch_size, seq_len, hidden); pass as-is (no extra unsqueeze)
        return self.generate(talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask)
    
    def generate(self, inputs_embeds: torch.Tensor, trailing_text_hiddens: torch.Tensor, tts_pad_embed: torch.Tensor, talker_attention_mask: torch.Tensor):
        pbar = tqdm(total=100, desc="Generating audio chunk")
        audio_codes = []
        talker_sampling_params = SamplingParams(temperature=1.0, max_tokens=1)
        predictor_sampling_params = SamplingParams(temperature=0.9, max_tokens=17)

        request_id = str(uuid.uuid4())
        generation_step = 0

        next_talker_embeds = inputs_embeds
        if next_talker_embeds.dim() == 2:
            next_talker_embeds = next_talker_embeds.unsqueeze(0)
        first_chunk_latency = None
        inner_chunk_latencies = []

        while True:
            start = time.time()
            self.talker_llm.add_request([next_talker_embeds], talker_sampling_params, request_id=request_id)

            _, _, outputs_all = self.talker_llm.step_with_outputs()
            if not outputs_all:
                self.talker_llm.clear_request(request_id)
                return audio_codes

            _, token_ids, hidden_states, _ = outputs_all[0]
            last_id = token_ids[-1]

            if last_id == 2150:
                self.talker_llm.clear_request(request_id)
                break

            last_id_hidden = self.input_embedding(torch.tensor([last_id], device=self.device)).unsqueeze(0)
            last_hidden_state = hidden_states.unsqueeze(0).unsqueeze(0)

            predictor_inputs_embeds = torch.cat((last_hidden_state, last_id_hidden), dim=1)

            predictor_outputs = self.predictor_llm.generate([predictor_inputs_embeds.unsqueeze(0)], predictor_sampling_params, use_tqdm=False)
            pred_token_ids = predictor_outputs[0]["token_ids"]
            codebook_ids = [last_id] + pred_token_ids
            audio_codes.append(codebook_ids)

            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [self.predictor_input_embeddings[i](torch.tensor([pred_token_ids[i]], device=self.device)).unsqueeze(0) for i in range(15)],
                dim=1,
            )
            next_talker_embeds = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hiddens.shape[1]:
                next_talker_embeds = next_talker_embeds + trailing_text_hiddens[:, generation_step].unsqueeze(1)
            else:
                next_talker_embeds = next_talker_embeds + tts_pad_embed

            generation_step += 1
            pbar.update(1)
            # if generation_step > 2:
            #     exit()

            end = time.time()
            chunk_latency = end - start
            
            if generation_step == 1:
                first_chunk_latency = chunk_latency
            else:
                inner_chunk_latencies.append(chunk_latency)
        
        pbar.close()
        
        if first_chunk_latency is not None:
            print(f"First chunk latency: {first_chunk_latency:.4f}s")
        if inner_chunk_latencies:
            avg_inner_latency = sum(inner_chunk_latencies) / len(inner_chunk_latencies)
            print(f"Inner chunk latency: {avg_inner_latency:.4f}s")
            
        return audio_codes

if __name__ == "__main__":
    interface = Qwen3TTSInterface(model_path="/work/weights/qwen3tts")
    print(f"Warm up...")
    audio_codes = interface.generate_custom_voice(text="Hi there this is a test.", language="English", speaker="Vivian")
    
    print(f"Generate...")
    
    start = time.time()
    
    audio_codes = interface.generate_custom_voice(text="Hi there, this is tsdocode, hope you are doing well.", language="English", speaker="Vivian")
    end = time.time()
    
    wavs, sr = tokenizer.decode([{"audio_codes": audio_codes}])
    sf.write("output_test.wav", wavs[0], sr)
    
    print(f"RTF: {(end - start)/(wavs[0].shape[-1]/sr)}")
    
    
    
    # import time
    # start = time.time()
    # audio_codes = interface.generate_custom_voice(text="Hi there, this is tsdocode, hope you are doing well.", language="English", speaker="Vivian")
    # end = time.time()
    
    # print(f"Audio codes: {audio_codes}")
    # print(f"Time taken: {end - start}s")