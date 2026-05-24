from __future__ import annotations

import json
import mimetypes
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000


def load_env(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env(ROOT / ".env")


CONFIG = {
    "port": int(os.environ.get("PORT", "5173")),
    "api_key": os.environ.get("SILICONFLOW_API_KEY", ""),
    "base_url": os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
    "chat_model": os.environ.get("SILICONFLOW_CHAT_MODEL", "THUDM/GLM-4-9B-0414"),
    "chat_fallbacks": [
        item.strip()
        for item in os.environ.get(
            "SILICONFLOW_CHAT_FALLBACKS",
            "Qwen/Qwen3.5-9B,deepseek-ai/DeepSeek-V4-Flash",
        ).split(",")
        if item.strip()
    ],
    "tts_model": os.environ.get("SILICONFLOW_TTS_MODEL", "FunAudioLLM/CosyVoice2-0.5B"),
    "tts_voice": os.environ.get(
        "SILICONFLOW_TTS_VOICE",
        "FunAudioLLM/CosyVoice2-0.5B:alex",
    ),
    "tts_format": os.environ.get("SILICONFLOW_TTS_FORMAT", "mp3"),
    "asr_model": os.environ.get(
        "LOCAL_ASR_MODEL",
        str(ROOT / "models" / "sherpa-onnx-paraformer-zh-2024-03-09" / "model.int8.onnx"),
    ),
    "asr_tokens": os.environ.get(
        "LOCAL_ASR_TOKENS",
        str(ROOT / "models" / "sherpa-onnx-paraformer-zh-2024-03-09" / "tokens.txt"),
    ),
    "punct_model": os.environ.get(
        "LOCAL_PUNCT_MODEL",
        str(
            ROOT
            / "models"
            / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"
            / "model.int8.onnx"
        ),
    ),
    "vad_model": os.environ.get(
        "LOCAL_VAD_MODEL",
        str(ROOT / "models" / "silero_vad.onnx"),
    ),
}


def log_backend(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)


class VoiceEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lock = threading.Lock()
        self.loaded = False
        self.load_error: Optional[str] = None
        self.np = None
        self.sherpa_onnx = None
        self.recognizer = None
        self.punctuation = None

    def status(self) -> Dict[str, Any]:
        missing = [
            path
            for path in [
                self.config["asr_model"],
                self.config["asr_tokens"],
                self.config["punct_model"],
                self.config["vad_model"],
            ]
            if not Path(path).is_file()
        ]

        return {
            "configured": not missing,
            "loaded": self.loaded,
            "error": self.load_error,
            "missing": missing,
            "sampleRate": SAMPLE_RATE,
            "asrModel": self.config["asr_model"],
            "punctModel": self.config["punct_model"],
            "vadModel": self.config["vad_model"],
        }

    def ensure_loaded(self) -> None:
        if self.loaded:
            return

        with self.lock:
            if self.loaded:
                return

            for key in ["asr_model", "asr_tokens", "punct_model", "vad_model"]:
                path = Path(self.config[key])
                if not path.is_file():
                    self.load_error = f"{key} not found: {path}"
                    raise RuntimeError(self.load_error)

            try:
                log_backend("[voice] models loading")
                import numpy as np
                import sherpa_onnx

                self.np = np
                self.sherpa_onnx = sherpa_onnx
                self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=self.config["asr_model"],
                    tokens=self.config["asr_tokens"],
                    num_threads=2,
                    sample_rate=SAMPLE_RATE,
                    feature_dim=80,
                    decoding_method="greedy_search",
                )
                punct_config = sherpa_onnx.OfflinePunctuationConfig(
                    model=sherpa_onnx.OfflinePunctuationModelConfig(
                        ct_transformer=self.config["punct_model"],
                    )
                )
                self.punctuation = sherpa_onnx.OfflinePunctuation(punct_config)
                self.loaded = True
                self.load_error = None
                log_backend("[voice] models ready")
            except Exception as exc:
                self.load_error = str(exc)
                log_backend(f"[voice] models failed error={exc}")
                raise

    def create_session(self, session_id: str) -> "VoiceSession":
        self.ensure_loaded()
        assert self.sherpa_onnx is not None

        vad_config = self.sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = self.config["vad_model"]
        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_silence_duration = 0.45
        vad_config.silero_vad.min_speech_duration = 0.15
        vad_config.sample_rate = SAMPLE_RATE
        vad = self.sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)
        return VoiceSession(session_id, self, vad)

    def recognize(self, samples: Any) -> str:
        self.ensure_loaded()
        assert self.recognizer is not None

        with self.lock:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            self.recognizer.decode_streams([stream])
            text = stream.result.text.strip()

            if text and self.punctuation is not None:
                text = self.punctuation.add_punctuation(text).strip()

            return text


class VoiceSession:
    def __init__(self, session_id: str, engine: VoiceEngine, vad: Any):
        self.session_id = session_id
        self.engine = engine
        self.vad = vad
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.lock = threading.Lock()
        self.chunk_count = 0
        self.total_audio_bytes = 0
        self.last_chunk_log_at = 0.0
        self.last_vad_state = "listening"

    def accept_pcm16(self, payload: bytes, flush: bool = False) -> Dict[str, Any]:
        self.updated_at = time.time()
        np = self.engine.np
        assert np is not None

        transcripts: List[str] = []
        speech_detected = False
        payload_len = len(payload)
        samples_len = 0
        rms = 0.0

        with self.lock:
            if payload:
                samples = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
                samples_len = len(samples)
                rms = float(np.sqrt(np.mean(samples * samples))) if samples_len else 0.0
                self.chunk_count += 1
                self.total_audio_bytes += payload_len
                for start in range(0, len(samples), 1600):
                    chunk = samples[start : start + 1600]
                    if len(chunk):
                        self.vad.accept_waveform(chunk)

            if flush:
                log_backend(f"[voice] flush session={self.session_id}")
                self.vad.flush()

            speech_detected = bool(self.vad.is_speech_detected())
            vad_state = "speech" if speech_detected else "listening"

            while not self.vad.empty():
                segment = self.vad.front
                segment_samples = np.asarray(segment.samples, dtype=np.float32)
                self.vad.pop()

                if len(segment_samples) < int(0.25 * SAMPLE_RATE):
                    log_backend(
                        "[voice] segment skipped "
                        f"session={self.session_id} seconds={len(segment_samples) / SAMPLE_RATE:.2f}"
                    )
                    continue

                started_at = time.time()
                log_backend(
                    "[voice] asr_start "
                    f"session={self.session_id} seconds={len(segment_samples) / SAMPLE_RATE:.2f}"
                )
                text = self.engine.recognize(segment_samples)
                elapsed_ms = int((time.time() - started_at) * 1000)
                if text:
                    transcripts.append(text)
                    log_voice_transcript(self.session_id, text, elapsed_ms)
                else:
                    log_backend(
                        f"[voice] asr_done session={self.session_id} elapsed_ms={elapsed_ms} text_empty=true"
                    )

            state = "transcript" if transcripts else vad_state
            self.maybe_log_chunk(payload_len, samples_len, rms, state)
            self.last_vad_state = state

        return {
            "state": state,
            "speechDetected": speech_detected,
            "transcripts": transcripts,
            "transcript": " ".join(transcripts).strip(),
        }

    def maybe_log_chunk(self, payload_len: int, samples_len: int, rms: float, state: str) -> None:
        if payload_len == 0:
            return

        now = time.time()
        should_log = (
            self.chunk_count <= 3
            or state != self.last_vad_state
            or now - self.last_chunk_log_at >= 1.0
        )
        if not should_log:
            return

        self.last_chunk_log_at = now
        log_backend(
            "[voice] chunk "
            f"session={self.session_id} count={self.chunk_count} bytes={payload_len} "
            f"samples={samples_len} rms={rms:.4f} state={state} total_bytes={self.total_audio_bytes}"
        )


VOICE_ENGINE = VoiceEngine(CONFIG)
VOICE_SESSIONS: Dict[str, VoiceSession] = {}
VOICE_SESSIONS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "ChatInNoisePython/0.1"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path_only == "/api/status":
            self.send_json(200, api_status())
            return

        self.serve_static()

    def do_HEAD(self) -> None:
        self.serve_static(head_only=True)

    def do_POST(self) -> None:
        if self.path_only.startswith("/api/voice/"):
            content_length = self.headers.get("Content-Length", "0")
            log_backend(
                f"[voice] post path={self.path_only} content_length={content_length}"
            )

        if self.path_only == "/api/chat":
            self.handle_chat()
            return

        if self.path_only == "/api/tts":
            self.handle_tts()
            return

        if self.path_only == "/api/voice/start":
            self.handle_voice_start()
            return

        if self.path_only == "/api/voice/chunk":
            self.handle_voice_chunk()
            return

        if self.path_only == "/api/voice/flush":
            self.handle_voice_flush()
            return

        if self.path_only == "/api/voice/reset":
            self.handle_voice_reset()
            return

        self.send_json(404, {"error": "Not found"})

    @property
    def path_only(self) -> str:
        return self.path.split("?", 1)[0]

    def handle_chat(self) -> None:
        if not CONFIG["api_key"]:
            self.send_json(500, {"error": "SILICONFLOW_API_KEY is not configured"})
            return

        body = self.read_json()
        prompt = body.get("message", "").strip() if isinstance(body.get("message"), str) else ""
        source = body.get("source", "unknown") if isinstance(body.get("source"), str) else "unknown"
        client_messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        if not prompt and not client_messages:
            self.send_json(400, {"error": "message is required"})
            return

        messages = normalize_messages(client_messages, prompt)
        failures = []
        log_backend(f"[chat] request source={source} user={safe_log_text(prompt)}")
        for model in unique([CONFIG["chat_model"], *CONFIG["chat_fallbacks"]]):
            try:
                started_at = time.time()
                payload = siliconflow_json(
                    "/chat/completions",
                    {
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "enable_thinking": False,
                        "max_tokens": 700,
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "frequency_penalty": 0.2,
                        "presence_penalty": 0,
                    },
                )
                reply = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if not reply:
                    raise RuntimeError("Empty model response")
                if is_suspicious_reply(reply):
                    raise RuntimeError(f"Suspicious model output: {safe_log_text(reply)}")

                elapsed_ms = int((time.time() - started_at) * 1000)
                log_backend(
                    "[chat] response "
                    f"source={source} model={model} elapsed_ms={elapsed_ms} "
                    f"reply={safe_log_text(reply)}"
                )
                self.send_json(
                    200,
                    {
                        "reply": reply,
                        "model": model,
                        "usage": payload.get("usage"),
                        "fallbackUsed": model != CONFIG["chat_model"],
                    },
                )
                return
            except Exception as exc:
                log_backend(f"[chat] model_failed model={model} error={exc}")
                failures.append(f"{model}: {exc}")

        self.send_json(
            502,
            {
                "error": "SiliconFlow chat request failed",
                "details": failures,
            },
        )

    def handle_tts(self) -> None:
        if not CONFIG["api_key"]:
            self.send_json(500, {"error": "SILICONFLOW_API_KEY is not configured"})
            return

        body = self.read_json()
        text = body.get("text", "").strip() if isinstance(body.get("text"), str) else ""
        if not text:
            self.send_json(400, {"error": "text is required"})
            return

        try:
            status, headers, audio = siliconflow_bytes(
                "/audio/speech",
                {
                    "model": CONFIG["tts_model"],
                    "input": text,
                    "voice": CONFIG["tts_voice"],
                    "response_format": CONFIG["tts_format"],
                    "sample_rate": 32000,
                    "speed": 1,
                    "gain": 0,
                    "stream": True,
                },
            )
        except urllib.error.HTTPError as exc:
            self.send_json(
                exc.code,
                {
                    "error": "SiliconFlow TTS request failed",
                    "details": exc.read().decode("utf-8", errors="replace"),
                },
            )
            return

        content_type = headers.get("Content-Type") or audio_content_type(CONFIG["tts_format"])
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(audio)

    def handle_voice_start(self) -> None:
        session_id = uuid.uuid4().hex
        log_backend(f"[voice] start_requested session={session_id}")

        try:
            session = VOICE_ENGINE.create_session(session_id)
        except Exception as exc:
            log_backend(f"[voice] start_failed session={session_id} error={exc}")
            self.send_json(500, {"error": "Voice engine failed to load", "details": str(exc)})
            return

        with VOICE_SESSIONS_LOCK:
            cleanup_voice_sessions()
            VOICE_SESSIONS[session_id] = session

        log_backend(f"[voice] start_ready session={session_id} sample_rate={SAMPLE_RATE}")
        self.send_json(
            200,
            {
                "sessionId": session_id,
                "sampleRate": SAMPLE_RATE,
                "status": VOICE_ENGINE.status(),
            },
        )

    def handle_voice_chunk(self) -> None:
        body = self.read_body()
        session = self.get_voice_session()
        if session is None:
            return

        result = session.accept_pcm16(body)
        self.send_json(200, result)

    def handle_voice_flush(self) -> None:
        session = self.get_voice_session()
        if session is None:
            return

        result = session.accept_pcm16(b"", flush=True)
        self.send_json(200, result)

    def handle_voice_reset(self) -> None:
        session_id = self.query_params().get("session")
        if session_id:
            with VOICE_SESSIONS_LOCK:
                VOICE_SESSIONS.pop(session_id, None)
            log_backend(f"[voice] reset session={session_id}")

        self.send_json(200, {"ok": True})

    def get_voice_session(self) -> Optional[VoiceSession]:
        session_id = self.query_params().get("session")
        if not session_id:
            self.send_json(400, {"error": "session is required"})
            return None

        with VOICE_SESSIONS_LOCK:
            session = VOICE_SESSIONS.get(session_id)

        if session is None:
            self.send_json(404, {"error": "voice session not found"})
            return None

        return session

    def serve_static(self, head_only: bool = False) -> None:
        request_path = self.path_only
        if request_path == "/":
            request_path = "/index.html"

        relative = urllib.parse.unquote(request_path).lstrip("/\\")

        file_path = (ROOT / relative).resolve()
        try:
            file_path.relative_to(ROOT)
        except ValueError:
            self.send_json(403, {"error": "Forbidden"})
            return

        if not file_path.is_file():
            self.send_json(404, {"error": "Not found"})
            return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        if file_path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif file_path.suffix in [".html", ".css", ".md"]:
            content_type = f"text/{file_path.suffix[1:]}; charset=utf-8"

        stat = file_path.stat()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        if not head_only:
            with file_path.open("rb") as f:
                self.wfile.write(f.read())

    def query_params(self) -> Dict[str, str]:
        if "?" not in self.path:
            return {}

        params = {}
        query = self.path.split("?", 1)[1]
        for item in query.split("&"):
            if not item:
                continue
            if "=" in item:
                key, value = item.split("=", 1)
            else:
                key, value = item, ""
            params[urllib.parse.unquote_plus(key)] = urllib.parse.unquote_plus(value)
        return params

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def read_json(self) -> Dict[str, Any]:
        raw = self.read_body()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)


def api_status() -> Dict[str, Any]:
    return {
        "hasKey": bool(CONFIG["api_key"]),
        "chatModel": CONFIG["chat_model"],
        "chatFallbacks": CONFIG["chat_fallbacks"],
        "ttsModel": CONFIG["tts_model"],
        "ttsVoice": CONFIG["tts_voice"],
        "ttsFormat": CONFIG["tts_format"],
        "voice": VOICE_ENGINE.status(),
    }


def normalize_messages(client_messages: List[Any], prompt: str) -> List[Dict[str, str]]:
    sanitized = []
    for message in client_messages[-8:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ["user", "assistant"] or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            sanitized.append({"role": role, "content": content})

    if prompt:
        last = sanitized[-1] if sanitized else None
        if last != {"role": "user", "content": prompt}:
            sanitized.append({"role": "user", "content": prompt})

    return [
        {
            "role": "system",
            "content": (
                "你是一个简洁、自然、可靠的中文语音对话助手。"
                "只用简体中文回答，准确优先；不要输出乱码、无意义音节、重复词、Markdown 表格或过长列表。"
                "回复要适合被直接朗读。"
            ),
        },
        *sanitized[-9:],
    ]


def siliconflow_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    status, _, body = siliconflow_bytes(path, payload)
    if status < 200 or status >= 300:
        raise RuntimeError(body.decode("utf-8", errors="replace"))
    return json.loads(body.decode("utf-8"))


def siliconflow_bytes(path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{CONFIG['base_url']}{path}",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {CONFIG['api_key']}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
    )

    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=90, context=context) as response:
        return response.status, dict(response.headers.items()), response.read()


def audio_content_type(format_name: str) -> str:
    if format_name == "wav":
        return "audio/wav"
    if format_name == "opus":
        return "audio/ogg"
    if format_name == "pcm":
        return "audio/L16"
    return "audio/mpeg"


def unique(items: List[str]) -> List[str]:
    result = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def log_voice_transcript(session_id: str, text: str, elapsed_ms: int) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(
        f"[{timestamp}] [voice] asr_done session={session_id} elapsed_ms={elapsed_ms} text={safe_log_text(text)}",
        flush=True,
    )


def safe_log_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) > limit:
        normalized = normalized[:limit] + "..."
    return json.dumps(normalized, ensure_ascii=False)


def is_suspicious_reply(text: str) -> bool:
    normalized = " ".join(text.split())
    if re.search(
        r"\b([A-Za-z0-9]{3,12})(?:[\s,，。、《》:：;；…\-]+\1\b){1,}",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(r"\b([A-Za-z]{2,3})\1+[A-Za-z]?\b", normalized, flags=re.IGNORECASE):
        return True

    latin_runs = re.findall(r"[A-Za-z]{4,}", normalized)
    if len(latin_runs) >= 2 and sum(len(item) for item in latin_runs) > max(12, len(normalized) * 0.18):
        return True

    return False


def cleanup_voice_sessions() -> None:
    now = time.time()
    expired = [
        session_id
        for session_id, session in VOICE_SESSIONS.items()
        if now - session.updated_at > 600
    ]
    for session_id in expired:
        VOICE_SESSIONS.pop(session_id, None)


def main() -> None:
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", CONFIG["port"]), Handler)
    print(f"ChatInNoise is running at http://localhost:{CONFIG['port']}", flush=True)
    print("Backend: Python + sherpa-onnx local VAD/ASR", flush=True)
    print("Voice debug logging: enabled", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
