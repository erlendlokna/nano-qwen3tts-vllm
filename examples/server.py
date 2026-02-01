"""FastAPI server for Qwen3-TTS text-to-speech generation.

Env:
  USE_ZMQ=1              - Use ZMQ (async engine loop + async queue).
  QWEN3_TTS_MODEL_PATH   - Model directory.
  HOST, PORT             - Server bind address.
"""

import asyncio
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Output format: 16-bit PCM at 24 kHz
TARGET_SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)

# Ensure log messages appear on console (works when run as uvicorn server:app or python server.py)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO)

# Lazy imports to avoid loading heavy models at module load
_interface = None
_tokenizer = None
_zmq_bridge = None
_decode_lock = threading.Lock()


def _use_zmq():
    """True if server should use ZMQ (background engine loop + queue-based generate)."""
    return os.environ.get("USE_ZMQ", "1").lower() in ("1", "true", "yes")


def get_interface():
    """Get or initialize the Qwen3TTSInterface (with or without ZMQ based on USE_ZMQ env)."""
    global _interface, _zmq_bridge
    if _interface is None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
        model_path = os.environ.get("QWEN3_TTS_MODEL_PATH", "/home/sang/work/weights/qwen3tts")
        if _use_zmq():
            from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
            _zmq_bridge = ZMQOutputBridge()
            _interface = Qwen3TTSInterface(
                model_path=model_path,
                zmq_bridge=_zmq_bridge,
                enforce_eager=False,
            )
        else:
            _interface = Qwen3TTSInterface(model_path=model_path)
    return _interface


def get_tokenizer():
    """Get or initialize the Qwen3TTSTokenizer for decoding audio codes."""
    global _tokenizer
    if _tokenizer is None:
        from qwen_tts import Qwen3TTSTokenizer
        _tokenizer = Qwen3TTSTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device_map="cuda:0",
        )
        _tokenizer.model.decoder = torch.compile(_tokenizer.model.decoder)
        
    return _tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up model and start ZMQ tasks when USE_ZMQ. Shutdown: stop ZMQ tasks and close bridge."""
    interface = get_interface()
    get_tokenizer()
    if _use_zmq() and interface.zmq_bridge is not None:
        await interface.start_zmq_tasks()
    yield
    if _use_zmq() and _interface is not None and _interface.zmq_bridge is not None:
        await _interface.stop_zmq_tasks()
        if _zmq_bridge is not None:
            _zmq_bridge.close()


app = FastAPI(
    title="Qwen3-TTS API",
    description="Text-to-speech generation using Qwen3-TTS with vLLM-style optimizations",
    version="0.1.0",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    """Request body for speech generation."""

    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="English", description="Language of the text")
    speaker: str = Field(default="Vivian", description="Speaker name")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def _float_to_pcm16(wav: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 PCM."""
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16)


def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample waveform to 24 kHz if needed."""
    if orig_sr == TARGET_SAMPLE_RATE:
        return wav
    n_orig = len(wav)
    n_new = int(round(n_orig * TARGET_SAMPLE_RATE / orig_sr))
    if n_new == 0:
        return wav
    indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
    return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)


def _decode_worker(
    codes_queue: queue.Queue,
    pcm_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run in thread: get audio_codes from codes_queue, decode, put PCM chunks into pcm_queue via call_soon_threadsafe."""
    prev_len_24k = 0
    try:
        tokenizer = get_tokenizer()
        while True:
            item = codes_queue.get()
            if item is None:
                loop.call_soon_threadsafe(pcm_queue.put_nowait, None)
                break
            audio_codes = item
            with _decode_lock:
                wav_list, sr = tokenizer.decode([{"audio_codes": audio_codes}])
            wav = wav_list[0]
            wav_24k = _resample_to_24k(wav, sr)
            pcm16 = _float_to_pcm16(wav_24k)
            chunk = pcm16[prev_len_24k:].tobytes()
            prev_len_24k = len(pcm16)
            if chunk:
                loop.call_soon_threadsafe(pcm_queue.put_nowait, chunk)
    except Exception as e:
        loop.call_soon_threadsafe(pcm_queue.put_nowait, e)


async def generate_speech_stream(request: SpeechRequest):
    """Async stream: generation loop pushes audio_codes to queue; decode worker runs in thread and yields PCM."""
    codes_queue: queue.Queue = queue.Queue()
    pcm_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    decode_thread = threading.Thread(
        target=_decode_worker,
        args=(codes_queue, pcm_queue, loop),
        daemon=True,
    )
    decode_thread.start()

    async def producer() -> None:
        audio_codes = []
        start_time = time.perf_counter()
        first_chunk_time = None
        last_chunk_time = None
        interface = get_interface()
        try:
            async for audio_code in interface.generate_custom_voice_async(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
            ):
                current_time = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_latency = current_time - start_time
                    first_chunk_time = current_time
                    last_chunk_time = current_time
                    logger.info(
                        "chunk #%d: %d codes, first_chunk_latency=%.3fs",
                        len(audio_codes) + 1,
                        len(audio_code),
                        first_chunk_latency,
                    )
                else:
                    inner_latency = current_time - last_chunk_time
                    last_chunk_time = current_time
                    logger.info(
                        "chunk #%d: %d codes, inner_chunk_latency=%.3fs",
                        len(audio_codes) + 1,
                        len(audio_code),
                        inner_latency,
                    )
                audio_codes.append(audio_code)
                if len(audio_codes) % 4 == 0:
                    codes_queue.put(list(audio_codes))
            
            codes_queue.put(list(audio_codes))
        finally:
            codes_queue.put(None)
        logger.info("speech stream producer done: %d chunks total", len(audio_codes))

    producer_task = asyncio.create_task(producer())

    try:
        while True:
            chunk = await pcm_queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
    finally:
        await producer_task


@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def generate_speech(request: SpeechRequest):
    """
    Generate speech from text.
    Returns raw PCM 16-bit mono at 24 kHz (audio/L16).
    Uses generate_custom_voice_async (requires USE_ZMQ=1).
    """
    try:
        return StreamingResponse(
            generate_speech_stream(request),
            media_type="audio/L16",
            headers={"Sample-Rate": str(TARGET_SAMPLE_RATE)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Qwen3-TTS API",
        "docs": "/docs",
        "health": "/health",
        "speech": "POST /v1/audio/speech (PCM16, 24 kHz mono)",
        "zmq": _use_zmq(),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    if _use_zmq():
        logger.info("Starting Qwen3-TTS API with ZMQ (async engine loop).")
    uvicorn.run(app, host=host, port=port)
