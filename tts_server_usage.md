# IndexTTS2 Tornado HTTP Server — `tts_server.py`

## 安装依赖

```bash
# 在项目虚拟环境中安装 tornado
.venv\Scripts\pip install tornado
# 或者用 uv
uv pip install tornado
```

---

## 启动服务

```bash
# 基本启动（默认端口 8000，监听所有网卡）
python tts_server.py

# 自定义端口 / 地址 / 模型目录 / 默认参考音频
python tts_server.py --port 8080 --host 0.0.0.0 --model_dir checkpoints --default_spk examples/voice_01.wav

# 指定配置文件路径
python tts_server.py --cfg_path checkpoints/config.yaml

# 启用 FP16 半精度推理（需 GPU 支持）
python tts_server.py --fp16

# 启用 BigVGAN 自定义 CUDA 核加速
python tts_server.py --cuda_kernel

# 启用 DeepSpeed 推理加速
python tts_server.py --deepspeed

# 启动时自动加载模型（兼容旧行为，启动即占用显存）
python tts_server.py --auto_load

# 默认启动（不自动加载模型，节省显存，通过接口按需加载）
python tts_server.py

# 查看所有参数说明
python tts_server.py --help
```

### 启动参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务监听地址 |
| `--port` | `8000` | 服务监听端口 |
| `--model_dir` | `checkpoints` | 模型权重文件目录 |
| `--cfg_path` | `checkpoints/config.yaml` | 模型配置文件路径 |
| `--default_spk` | `examples/voice_01.wav` | 请求未指定参考音频时的默认音频 |
| `--fp16` | `false` | 启用 FP16 半精度推理 |
| `--cuda_kernel` | `false` | 启用 BigVGAN 自定义 CUDA 核 |
| `--deepspeed` | `false` | 启用 DeepSpeed 加速 |
| `--auto_load` | `false` | 启动时自动加载模型（默认不加载，通过 `POST /model/load` 按需加载） |

---

## API 接口

### 模型按需加载/卸载

默认启动时不加载模型，需通过以下接口按需管理，避免长期占用 GPU 显存。

---

### `GET /model/status` — 查询模型状态

```bash
curl http://localhost:8000/model/status
```

响应示例：
```json
{"loaded": false, "model_version": null, "device": null}
{"loaded": true,  "model_version": "2.0", "device": "cuda:0"}
```

---

### `POST /model/load` — 加载模型

请求体所有字段均可选，省略则使用启动参数默认值：

```bash
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "checkpoints",
    "cfg_path":  "checkpoints/config.yaml",
    "fp16":        true,
    "cuda_kernel": false,
    "deepspeed":   false
  }'
```

响应：
```json
{"status": "loaded", "model_version": "2.0", "device": "cuda:0", "elapsed": 18.5}
```
若模型已加载，直接返回：
```json
{"status": "already_loaded", "model_version": "2.0", "device": "cuda:0"}
```

---

### `POST /model/unload` — 卸载模型（释放显存）

```bash
curl -X POST http://localhost:8000/model/unload
```

响应：
```json
{"status": "unloaded"}
```
若模型未加载：
```json
{"status": "not_loaded"}
```

---

### `GET /health` — 健康检查

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": false, "model_version": null}
```

---

### `POST /tts` — 生成语音

支持两种请求方式，按场景选择：

| 方式 | Content-Type | 参考音频来源 |
|------|-------------|------------|
| 方式一 | `multipart/form-data` | **直接上传音频文件**，或填写服务器路径 |
| 方式二 | `application/json` | 服务器本地文件路径 |

---

#### 方式一：multipart/form-data（支持文件上传）

适合前端表单或需要动态指定参考音频的场景，可直接将音频文件上传给服务器，无需提前部署到服务器目录。

**参考音频来源优先级（三选一）：**
1. 上传文件字段 `spk_audio_prompt`（最高优先级）
2. 表单字段 `spk_audio_prompt_path`（服务器本地路径）
3. 服务启动时的 `--default_spk` 默认音频（兜底）

**表单字段说明：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | string | — | **必填**，待合成文本 |
| `spk_audio_prompt` | **file** | — | 上传的参考音频文件（wav / mp3 等） |
| `spk_audio_prompt_path` | string | — | 服务器本地参考音频路径（与上传二选一） |
| `emo_audio_prompt` | **file** | — | 上传的情绪参考音频文件 |
| `emo_audio_prompt_path` | string | — | 服务器本地情绪参考音频路径（与上传二选一） |
| `emo_alpha` | float | `1.0` | 情绪混合强度 |
| `emo_vector` | string | — | JSON 序列化的 8 维情绪向量，如 `"[0.1,0.2,0,0,0,0,0,0]"` |
| `use_emo_text` | bool | `false` | 是否用文本自动推断情绪（`true`/`false`） |
| `emo_text` | string | — | 情绪描述文本 |
| `use_random` | bool | `false` | 是否使用随机情绪索引 |
| `interval_silence` | int | `200` | 分段间静音时长（ms） |
| `verbose` | bool | `false` | 是否打印详细推理日志 |
| `max_text_tokens_per_segment` | int | `120` | 每段最大 token 数 |
| `do_sample` | bool | `true` | 是否使用采样解码 |
| `top_p` | float | `0.8` | nucleus sampling 参数 |
| `top_k` | int | `30` | top-k 采样参数 |
| `temperature` | float | `0.8` | 采样温度 |
| `length_penalty` | float | `0.0` | 长度惩罚 |
| `num_beams` | int | `3` | beam search 宽度 |
| `repetition_penalty` | float | `10.0` | 重复惩罚系数 |
| `max_mel_tokens` | int | `1500` | 最大 mel token 数 |

> **注意**：表单中所有非文件字段均以字符串传入，服务端会自动转换为对应类型。

---

#### 方式二：application/json（使用服务器本地路径）

适合服务端之间调用，或参考音频已预置在服务器上的场景。

**请求体字段说明：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | string | — | **必填**，待合成文本 |
| `spk_audio_prompt` | string | `--default_spk` | 服务器本地参考音频路径 |
| `emo_audio_prompt` | string | `null` | 服务器本地情绪参考音频路径 |
| `emo_alpha` | float | `1.0` | 情绪混合强度 |
| `emo_vector` | list[float] | `null` | 8 维情绪向量 |
| `use_emo_text` | bool | `false` | 是否用文本自动推断情绪 |
| `emo_text` | string | `null` | 情绪描述文本 |
| `use_random` | bool | `false` | 是否使用随机情绪索引 |
| `interval_silence` | int | `200` | 分段间静音时长（ms） |
| `verbose` | bool | `false` | 是否打印详细推理日志 |
| `max_text_tokens_per_segment` | int | `120` | 每段最大 token 数 |
| `do_sample` | bool | `true` | 是否使用采样解码 |
| `top_p` | float | `0.8` | nucleus sampling 参数 |
| `top_k` | int | `30` | top-k 采样参数 |
| `temperature` | float | `0.8` | 采样温度 |
| `length_penalty` | float | `0.0` | 长度惩罚 |
| `num_beams` | int | `3` | beam search 宽度 |
| `repetition_penalty` | float | `10.0` | 重复惩罚系数 |
| `max_mel_tokens` | int | `1500` | 最大 mel token 数 |

---

#### 响应

| 响应头 | 说明 |
|--------|------|
| `Content-Type` | `audio/wav` |
| `Content-Disposition` | `attachment; filename="tts_xxxxxxxx.wav"` |
| `Content-Length` | WAV 文件字节数 |
| `X-Inference-Time` | 推理耗时，如 `3.142s` |

Body 为 WAV 音频二进制数据，可直接保存为 `.wav` 文件播放。

---

## 调用示例

### curl — 上传音频文件（multipart/form-data）

```bash
# 上传本地音频文件作为参考音频
curl -X POST http://localhost:8000/tts \
  -F "text=你好，这是一段测试语音" \
  -F "spk_audio_prompt=@/path/to/my_voice.wav" \
  --output output.wav

# 同时指定情绪参考音频 + 采样参数
curl -X POST http://localhost:8000/tts \
  -F "text=Translate for me, what is a surprise!" \
  -F "spk_audio_prompt=@examples/voice_01.wav" \
  -F "emo_audio_prompt=@examples/voice_02.wav" \
  -F "emo_alpha=0.8" \
  -F "temperature=0.7" \
  --output output.wav

# 不上传文件，使用服务器本地路径
curl -X POST http://localhost:8000/tts \
  -F "text=欢迎使用语音合成服务" \
  -F "spk_audio_prompt_path=examples/voice_03.wav" \
  --output output.wav
```

### curl — 使用 JSON（application/json）

```bash
# 最简调用，使用默认参考音频
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "欢迎使用IndexTTS2语音合成服务！"}' \
  --output output.wav

# 指定参考音频 + 情绪文本 + 解码参数
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Translate for me, what is a surprise!",
    "spk_audio_prompt": "examples/voice_01.wav",
    "use_emo_text": true,
    "temperature": 0.7,
    "top_p": 0.85
  }' \
  --output output.wav
```

### Python — 上传音频文件

```python
import requests

# 上传本地音频文件
with open("my_voice.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/tts",
        files={"spk_audio_prompt": ("my_voice.wav", f, "audio/wav")},
        data={
            "text": "你好，这是一段测试语音",
            "temperature": "0.8",
            "top_p": "0.85",
        },
        timeout=120,
    )

resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
print("推理耗时:", resp.headers.get("X-Inference-Time"))
```

### Python — 使用 JSON（服务器本地路径）

```python
import requests

resp = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "Translate for me, what is a surprise!",
        "spk_audio_prompt": "examples/voice_01.wav",
        "use_emo_text": True,
        "temperature": 0.8,
        "repetition_penalty": 10.0,
    },
    timeout=120,
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
print("推理耗时:", resp.headers.get("X-Inference-Time"))
```

### Python — 健康检查

```python
import requests

resp = requests.get("http://localhost:8000/health", timeout=5)
print(resp.json())  # {"status": "ok", "model_version": "..."}
```
