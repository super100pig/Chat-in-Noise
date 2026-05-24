# ChatInNoise

本地网页语音对话原型：

- 前端负责播放背景循环音、AI TTS 音频和麦克风采集。
- Python 后端代理硅基流动 Chat/TTS，不把 API key 暴露给浏览器。
- 本地 VAD/ASR 使用 `sherpa-onnx`，VAD 模型为 Silero ONNX，ASR/Punct 模型放在 `models/`。

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

模型文件：

```text
models/silero_vad.onnx
models/sherpa-onnx-paraformer-zh-2024-03-09/model.int8.onnx
models/sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt
models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx
```

启动：

```powershell
.\.venv\Scripts\python server.py
```

打开：

```text
http://localhost:5173
```
