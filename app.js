const defaultConfig = {
  modelEndpoint: "/api/chat",
  ttsEndpoint: "/api/tts",
  statusEndpoint: "/api/status",
  voiceStartEndpoint: "/api/voice/start",
  voiceChunkEndpoint: "/api/voice/chunk",
  voiceFlushEndpoint: "/api/voice/flush",
  voiceResetEndpoint: "/api/voice/reset",
  backgroundAudioSrc: "assets/background-loop.mp3",
  backgroundVolume: 0.58,
};

const config = {
  ...defaultConfig,
  ...(window.ChatInNoiseConfig || {}),
};

const state = {
  aiVolume: 0.7,
  activeAiObjectUrl: null,
  backgroundStarted: false,
  messages: [],
  voiceActive: false,
  voicePaused: false,
  voiceStarting: false,
  voiceSending: false,
  voiceSessionId: null,
  voiceTargetSampleRate: 16000,
  voiceQueue: [],
  voicePendingChunks: [],
  voicePendingLength: 0,
  backgroundChecked: false,
  backgroundAvailable: false,
  backgroundCheckPromise: null,
  voiceContext: null,
  voiceStream: null,
  voiceSource: null,
  voiceProcessor: null,
};

const els = {
  promptForm: document.querySelector("#promptForm"),
  promptInput: document.querySelector("#promptInput"),
  messageList: document.querySelector("#messageList"),
  modelStatus: document.querySelector("#modelStatus"),
  ambientStatus: document.querySelector("#ambientStatus"),
  ambientButton: document.querySelector("#ambientButton"),
  voiceState: document.querySelector("#voiceState"),
  replyLatency: document.querySelector("#replyLatency"),
  volumeUp: document.querySelector("#volumeUp"),
  volumeDown: document.querySelector("#volumeDown"),
  volumePercent: document.querySelector("#volumePercent"),
  volumeFill: document.querySelector("#volumeFill"),
  voiceToggleButton: document.querySelector("#voiceToggleButton"),
  voiceToggleLabel: document.querySelector("#voiceToggleLabel"),
  stopVoiceButton: document.querySelector("#stopVoiceButton"),
  backgroundAudio: document.querySelector("#backgroundAudio"),
  aiAudio: document.querySelector("#aiAudio"),
  modelEndpointStatus: document.querySelector("#modelEndpointStatus"),
  ttsEndpointStatus: document.querySelector("#ttsEndpointStatus"),
  voiceEndpointStatus: document.querySelector("#voiceEndpointStatus"),
  backgroundEndpointStatus: document.querySelector("#backgroundEndpointStatus"),
};

function init() {
  els.backgroundAudio.volume = config.backgroundVolume;
  els.aiAudio.volume = state.aiVolume;

  els.modelEndpointStatus.textContent = config.modelEndpoint ? "连接中" : "未配置";
  els.ttsEndpointStatus.textContent = config.ttsEndpoint ? "连接中" : "浏览器预览";
  els.backgroundEndpointStatus.textContent = config.backgroundAudioSrc ? "检查中" : "未配置";

  updateVolumeUi();
  bindEvents();
  prepareBackgroundAudio();
  refreshServerStatus();
}

function bindEvents() {
  els.promptForm.addEventListener("submit", handlePromptSubmit);
  els.ambientButton.addEventListener("click", ensureBackgroundLoop);
  els.volumeUp.addEventListener("click", () => adjustAiVolume(0.1));
  els.volumeDown.addEventListener("click", () => adjustAiVolume(-0.1));
  els.voiceToggleButton.addEventListener("click", toggleVoiceConversation);
  els.stopVoiceButton.addEventListener("click", stopAiVoice);

  els.aiAudio.addEventListener("play", () => {
    setVoiceState("正在播放");
    pauseVoiceInput("AI 播放中");
  });
  els.aiAudio.addEventListener("ended", () => {
    setVoiceState("播放完成");
    resumeVoiceInputSoon();
  });
  els.aiAudio.addEventListener("error", () => {
    setVoiceState("语音播放失败");
    resumeVoiceInputSoon();
  });

  els.backgroundAudio.addEventListener("play", () => {
    state.backgroundStarted = true;
    els.ambientStatus.textContent = "背景声循环中";
    els.ambientButton.classList.add("is-active");
    els.ambientButton.querySelector("span:last-child").textContent = "声场循环";
  });

  els.backgroundAudio.addEventListener("error", () => {
    state.backgroundStarted = false;
    els.ambientStatus.textContent = "等待背景音";
    els.ambientButton.classList.remove("is-active");
  });
}

async function handlePromptSubmit(event) {
  event.preventDefault();

  const prompt = els.promptInput.value.trim();
  if (!prompt) {
    els.promptInput.focus();
    return;
  }

  await submitPrompt(prompt, "manual");
}

async function submitPrompt(prompt, source = "manual") {
  appendMessage("user", "我", prompt);
  state.messages.push({ role: "user", content: prompt });
  els.promptInput.value = "";
  els.replyLatency.textContent = "请求中";
  els.modelStatus.textContent = "模型生成中";
  setVoiceState("等待模型回复");

  ensureBackgroundLoop();

  const startedAt = performance.now();

  try {
    const reply = await requestModelReply(prompt, source);
    const elapsed = Math.max(1, Math.round(performance.now() - startedAt));
    els.replyLatency.textContent = `${elapsed}ms`;
    els.modelStatus.textContent = "模型已回复";
    appendMessage("ai", "AI", reply);
    state.messages.push({ role: "assistant", content: reply });
    state.messages = state.messages.slice(-8);
    await speakReply(reply);
  } catch (error) {
    els.replyLatency.textContent = "请求失败";
    els.modelStatus.textContent = "接口异常";
    setVoiceState("回复失败");
    appendMessage("ai", "AI", "请求失败，请检查模型接口配置。");
    resumeVoiceInputSoon();
    console.error(error);
  }
}

async function requestModelReply(prompt, source = "manual") {
  if (!config.modelEndpoint) {
    await delay(650);
    return `收到：「${prompt}」。这里会替换成大模型 API 的真实回复，然后进入 AI 语音通道播放。`;
  }

  const response = await fetch(config.modelEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: prompt,
      messages: state.messages,
      source,
    }),
  });

  if (!response.ok) {
    const payload = await readJsonSafe(response);
    const message = payload?.details || payload?.error || response.statusText;
    throw new Error(`Model API failed with ${response.status}: ${message}`);
  }

  const payload = await response.json();
  return (
    payload.reply ||
    payload.text ||
    payload.content ||
    payload.output_text ||
    payload.message ||
    JSON.stringify(payload)
  );
}

async function speakReply(text) {
  stopAiVoice(false);
  setVoiceState("生成语音中");

  if (config.ttsEndpoint) {
    const response = await fetch(config.ttsEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const payload = await readJsonSafe(response);
      const message = payload?.details || payload?.error || response.statusText;
      throw new Error(`TTS API failed with ${response.status}: ${message}`);
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    state.activeAiObjectUrl = audioUrl;
    els.aiAudio.src = audioUrl;
    els.aiAudio.volume = state.aiVolume;
    await els.aiAudio.play();
    return;
  }

  if (!("speechSynthesis" in window)) {
    setVoiceState("浏览器不支持语音");
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "zh-CN";
  utterance.rate = 0.96;
  utterance.pitch = 1;
  utterance.volume = state.aiVolume;
  utterance.onstart = () => {
    setVoiceState("正在播放");
    pauseVoiceInput("AI 播放中");
  };
  utterance.onend = () => {
    setVoiceState("播放完成");
    resumeVoiceInputSoon();
  };
  utterance.onerror = () => {
    setVoiceState("语音播放失败");
    resumeVoiceInputSoon();
  };
  window.speechSynthesis.speak(utterance);
}

async function refreshServerStatus() {
  if (!config.statusEndpoint) {
    return;
  }

  try {
    const response = await fetch(config.statusEndpoint);
    if (!response.ok) {
      throw new Error(`Status request failed with ${response.status}`);
    }

    const payload = await response.json();
    els.modelEndpointStatus.textContent = payload.hasKey
      ? payload.chatModel || "已配置"
      : "缺少密钥";
    els.ttsEndpointStatus.textContent = payload.hasKey
      ? `${payload.ttsModel || "已配置"} / ${voiceName(payload.ttsVoice)}`
      : "缺少密钥";
    els.voiceEndpointStatus.textContent = payload.voice?.configured
      ? payload.voice.loaded
        ? "已加载"
        : "可启动"
      : "模型缺失";
    els.modelStatus.textContent = payload.hasKey ? "模型待机" : "密钥未配置";
  } catch (error) {
    els.modelEndpointStatus.textContent = "服务未启动";
    els.ttsEndpointStatus.textContent = "服务未启动";
    els.voiceEndpointStatus.textContent = "服务未启动";
    els.modelStatus.textContent = "等待服务";
    console.warn(error);
  }
}

async function toggleVoiceConversation() {
  if (state.voiceActive || state.voiceStarting) {
    await stopVoiceConversation();
    return;
  }

  await startVoiceConversation();
}

async function startVoiceConversation() {
  if (!navigator.mediaDevices?.getUserMedia) {
    els.voiceEndpointStatus.textContent = "浏览器不支持麦克风";
    return;
  }

  state.voiceStarting = true;
  updateVoiceUi("初始化中");
  console.info("[voice] start requested");

  try {
    const startResponse = await fetch(config.voiceStartEndpoint, {
      method: "POST",
    });

    if (!startResponse.ok) {
      const payload = await readJsonSafe(startResponse);
      throw new Error(payload?.details || payload?.error || "本地语音服务启动失败");
    }

    const startPayload = await startResponse.json();
    state.voiceSessionId = startPayload.sessionId;
    state.voiceTargetSampleRate = startPayload.sampleRate || 16000;
    console.info("[voice] backend session", state.voiceSessionId);

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: true,
      },
    });
    console.info("[voice] microphone granted");

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (event) => {
      event.outputBuffer.getChannelData(0).fill(0);

      if (!state.voiceActive || state.voicePaused) {
        return;
      }

      const input = event.inputBuffer.getChannelData(0);
      const pcm = downsampleToInt16(input, context.sampleRate, state.voiceTargetSampleRate);
      enqueueVoiceChunk(pcm);
    };

    source.connect(processor);
    processor.connect(context.destination);

    state.voiceContext = context;
    state.voiceStream = stream;
    state.voiceSource = source;
    state.voiceProcessor = processor;
    state.voiceActive = true;
    state.voicePaused = false;
    state.voiceQueue = [];
    state.voicePendingChunks = [];
    state.voicePendingLength = 0;
    state.voiceStarting = false;

    await context.resume();
    updateVoiceUi("聆听中");
    console.info("[voice] listening", {
      inputSampleRate: context.sampleRate,
      targetSampleRate: state.voiceTargetSampleRate,
    });
  } catch (error) {
    state.voiceStarting = false;
    await stopVoiceConversation();
    els.voiceEndpointStatus.textContent = "启动失败";
    console.error(error);
  }
}

async function stopVoiceConversation() {
  const sessionId = state.voiceSessionId;

  state.voiceActive = false;
  state.voicePaused = false;
  state.voiceStarting = false;
  state.voiceQueue = [];
  state.voicePendingChunks = [];
  state.voicePendingLength = 0;

  if (state.voiceProcessor) {
    state.voiceProcessor.disconnect();
    state.voiceProcessor.onaudioprocess = null;
  }

  if (state.voiceSource) {
    state.voiceSource.disconnect();
  }

  if (state.voiceStream) {
    state.voiceStream.getTracks().forEach((track) => track.stop());
  }

  if (state.voiceContext) {
    await state.voiceContext.close().catch(() => {});
  }

  state.voiceContext = null;
  state.voiceStream = null;
  state.voiceSource = null;
  state.voiceProcessor = null;
  state.voiceSessionId = null;

  if (sessionId) {
    fetch(`${config.voiceResetEndpoint}?session=${encodeURIComponent(sessionId)}`, {
      method: "POST",
    }).catch(() => {});
  }

  updateVoiceUi("已关闭");
}

function enqueueVoiceChunk(pcm) {
  if (!pcm.byteLength || !state.voiceSessionId) {
    return;
  }

  state.voicePendingChunks.push(pcm);
  state.voicePendingLength += pcm.length;

  if (state.voicePendingLength < Math.round(state.voiceTargetSampleRate * 0.5)) {
    return;
  }

  const packet = new Int16Array(state.voicePendingLength);
  let offset = 0;
  for (const chunk of state.voicePendingChunks) {
    packet.set(chunk, offset);
    offset += chunk.length;
  }

  state.voicePendingChunks = [];
  state.voicePendingLength = 0;
  state.voiceQueue.push(packet);
  if (state.voiceQueue.length === 1) {
    console.debug("[voice] queued pcm packet", packet.byteLength);
  }

  if (state.voiceQueue.length > 18) {
    state.voiceQueue.splice(0, state.voiceQueue.length - 18);
  }

  processVoiceQueue();
}

async function processVoiceQueue() {
  if (state.voiceSending || !state.voiceActive || !state.voiceSessionId) {
    return;
  }

  state.voiceSending = true;

  try {
    while (
      state.voiceActive &&
      !state.voicePaused &&
      state.voiceSessionId &&
      state.voiceQueue.length
    ) {
      const pcm = state.voiceQueue.shift();
      const body = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength);
      const response = await fetch(
        `${config.voiceChunkEndpoint}?session=${encodeURIComponent(state.voiceSessionId)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/octet-stream",
          },
          body,
        },
      );

      if (!response.ok) {
        const payload = await readJsonSafe(response);
        if (response.status === 404) {
          await stopVoiceConversation();
          els.voiceEndpointStatus.textContent = "语音会话已过期";
          return;
        }
        throw new Error(payload?.error || "语音识别请求失败");
      }

      const payload = await response.json();
      await handleVoiceResult(payload);
    }
  } catch (error) {
    els.voiceEndpointStatus.textContent = "识别异常";
    console.error(error);
  } finally {
    state.voiceSending = false;
  }
}

async function handleVoiceResult(payload) {
  if (payload.transcript) {
    const transcript = payload.transcript.trim();
    if (!transcript) {
      return;
    }

    pauseVoiceInput("识别完成");
    setVoiceState(`听写：${transcript}`);
    await submitPrompt(transcript, "voice");
    return;
  }

  if (payload.speechDetected) {
    updateVoiceUi("检测到人声");
  } else if (state.voiceActive && !state.voicePaused) {
    updateVoiceUi("聆听中");
  }
}

function pauseVoiceInput(label) {
  if (!state.voiceActive) {
    return;
  }

  state.voicePaused = true;
  state.voiceQueue = [];
  state.voicePendingChunks = [];
  state.voicePendingLength = 0;
  updateVoiceUi(label);
}

function resumeVoiceInputSoon() {
  if (!state.voiceActive) {
    return;
  }

  window.setTimeout(() => {
    if (!state.voiceActive) {
      return;
    }

    state.voicePaused = false;
    updateVoiceUi("聆听中");
    processVoiceQueue();
  }, 700);
}

async function ensureBackgroundLoop() {
  if (state.backgroundStarted) {
    return;
  }

  await prepareBackgroundAudio();

  if (!config.backgroundAudioSrc || !state.backgroundAvailable) {
    els.ambientStatus.textContent = "背景音未配置";
    return;
  }

  try {
    els.backgroundAudio.volume = config.backgroundVolume;
    await els.backgroundAudio.play();
  } catch (error) {
    els.ambientStatus.textContent = "点击启动背景声";
    console.warn(error);
  }
}

async function prepareBackgroundAudio() {
  if (state.backgroundChecked) {
    return state.backgroundAvailable;
  }

  if (state.backgroundCheckPromise) {
    return state.backgroundCheckPromise;
  }

  if (!config.backgroundAudioSrc) {
    state.backgroundChecked = true;
    state.backgroundAvailable = false;
    els.backgroundEndpointStatus.textContent = "未配置";
    els.ambientStatus.textContent = "背景音未配置";
    return false;
  }

  state.backgroundCheckPromise = fetch(config.backgroundAudioSrc, {
    method: "HEAD",
    cache: "no-store",
  })
    .then((response) => {
      state.backgroundChecked = true;
      state.backgroundAvailable = response.ok;

      if (response.ok) {
        els.backgroundAudio.src = config.backgroundAudioSrc;
        els.backgroundEndpointStatus.textContent = config.backgroundAudioSrc;
      } else {
        els.backgroundEndpointStatus.textContent = "文件缺失";
        els.ambientStatus.textContent = "背景音缺失";
      }

      return state.backgroundAvailable;
    })
    .catch((error) => {
      state.backgroundChecked = true;
      state.backgroundAvailable = false;
      els.backgroundEndpointStatus.textContent = "检查失败";
      els.ambientStatus.textContent = "背景音缺失";
      console.warn(error);
      return false;
    })
    .finally(() => {
      state.backgroundCheckPromise = null;
    });

  return state.backgroundCheckPromise;
}

function stopAiVoice(resumeAfterStop = true) {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  els.aiAudio.pause();
  els.aiAudio.currentTime = 0;

  if (state.activeAiObjectUrl) {
    URL.revokeObjectURL(state.activeAiObjectUrl);
    state.activeAiObjectUrl = null;
  }

  setVoiceState("已停止");

  if (resumeAfterStop) {
    resumeVoiceInputSoon();
  }
}

function adjustAiVolume(delta) {
  state.aiVolume = clamp(state.aiVolume + delta, 0, 1);
  els.aiAudio.volume = state.aiVolume;
  updateVolumeUi();
}

function updateVolumeUi() {
  const percent = Math.round(state.aiVolume * 100);
  els.volumePercent.textContent = `${percent}%`;
  els.volumeFill.style.width = `${percent}%`;
}

function updateVoiceUi(label) {
  els.voiceEndpointStatus.textContent = label;
  els.voiceToggleButton.classList.toggle("is-active", state.voiceActive || state.voiceStarting);
  els.voiceToggleLabel.textContent = state.voiceActive || state.voiceStarting ? "关闭语音" : "语音对话";
}

function setVoiceState(label) {
  els.voiceState.textContent = label;
  document.body.classList.toggle("is-speaking", label === "正在播放");
}

function appendMessage(type, role, text) {
  const message = document.createElement("article");
  message.className = `message ${type}`;

  const roleEl = document.createElement("span");
  roleEl.className = "message-role";
  roleEl.textContent = role;

  const textEl = document.createElement("p");
  textEl.textContent = text;

  message.append(roleEl, textEl);
  els.messageList.append(message);
  els.messageList.scrollTop = els.messageList.scrollHeight;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function readJsonSafe(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function voiceName(voice) {
  return String(voice || "")
    .split(":")
    .pop();
}

function downsampleToInt16(input, inputSampleRate, outputSampleRate) {
  if (inputSampleRate === outputSampleRate) {
    return floatToInt16(input);
  }

  const ratio = inputSampleRate / outputSampleRate;
  const outputLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Int16Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    let count = 0;

    for (let j = start; j < end; j += 1) {
      sum += input[j];
      count += 1;
    }

    const sample = count ? sum / count : input[start] || 0;
    output[i] = floatSampleToInt16(sample);
  }

  return output;
}

function floatToInt16(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    output[i] = floatSampleToInt16(input[i]);
  }
  return output;
}

function floatSampleToInt16(sample) {
  const clamped = clamp(sample, -1, 1);
  return clamped < 0 ? clamped * 32768 : clamped * 32767;
}

init();
