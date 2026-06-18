"""
IndexTTS2 Tornado HTTP 语音合成服务

功能说明:
    基于 Tornado 框架封装 IndexTTS2 模型，提供 HTTP 接口，
    接收文本和参考音频，返回合成的 WAV 语音文件。
    支持通过接口动态加载/卸载模型，以按需占用显存。

启动方式:
    python tts_server.py [--port 8000] [--host 0.0.0.0] [--model_dir checkpoints]
                         [--fp16] [--cuda_kernel] [--deepspeed] [--auto_load]

API 接口说明:
    POST /tts
        支持两种请求方式，可按需选择：

        ── 方式一：multipart/form-data（支持直接上传音频文件）──────────────────
        Content-Type: multipart/form-data
        表单字段:
            text                        string  必填，待合成文本
            spk_audio_prompt            file    可选，上传的参考音频文件（wav/mp3 等）
            spk_audio_prompt_path       string  可选，服务器本地参考音频路径（与上传二选一）
            emo_audio_prompt            file    可选，上传的情绪参考音频文件
            emo_audio_prompt_path       string  可选，服务器本地情绪参考音频路径（与上传二选一）
            emo_alpha / emo_vector / use_emo_text / emo_text / use_random /
            interval_silence / verbose / max_text_tokens_per_segment /
            do_sample / top_p / top_k / temperature / length_penalty /
            num_beams / repetition_penalty / max_mel_tokens
                                        同 JSON 方式各字段含义，以字符串形式传入

        ── 方式二：application/json（使用服务器本地音频路径）──────────────────
        Content-Type: application/json
        请求体:
            {
                "text": "要合成的文本",                          // 必填
                "spk_audio_prompt": "examples/voice_01.wav",   // 可选，服务器本地参考音频路径
                "emo_audio_prompt": null,                       // 可选，情绪参考音频路径
                "emo_alpha": 1.0,                               // 可选，情绪混合强度
                "emo_vector": null,                             // 可选，8维情绪向量列表
                "use_emo_text": false,                         // 可选，是否用文本自动推断情绪
                "emo_text": null,                               // 可选，情绪描述文本
                "use_random": false,                            // 可选，是否随机情绪索引
                "interval_silence": 200,                        // 可选，分段间静音时长(ms)
                "verbose": false,                               // 可选，是否打印详细推理日志
                "max_text_tokens_per_segment": 120,            // 可选，每段最大 token 数
                "do_sample": true,                             // 可选，是否使用采样解码
                "top_p": 0.8,                                  // 可选，nucleus sampling 参数
                "top_k": 30,                                   // 可选，top-k 采样参数
                "temperature": 0.8,                            // 可选，采样温度
                "length_penalty": 0.0,                         // 可选，长度惩罚
                "num_beams": 3,                                 // 可选，beam search 宽度
                "repetition_penalty": 10.0,                    // 可选，重复惩罚系数
                "max_mel_tokens": 1500                         // 可选，最大 mel token 数
            }
        响应:
            Content-Type: audio/wav
            Body: WAV 音频二进制数据
            X-Inference-Time: 推理耗时（秒）

    GET /health
        响应 (JSON): {"status": "ok", "model_version": "...", "model_loaded": true/false}
        用于服务健康检查/存活探测

    GET /model/status
        响应 (JSON): {"loaded": true/false, "model_version": "...", "device": "cuda:0"}
        查询模型当前加载状态

    POST /model/load
        Content-Type: application/json（所有字段均可选，省略则使用启动参数默认值）
        请求体:
            {
                "model_dir": "checkpoints",     // 可选，模型目录
                "cfg_path": "checkpoints/config.yaml",  // 可选，配置文件路径
                "fp16": true,                   // 可选，是否使用 FP16
                "cuda_kernel": false,           // 可选，是否使用 CUDA 自定义核
                "deepspeed": false              // 可选，是否使用 DeepSpeed
            }
        响应 (JSON): {"status": "loaded", "model_version": "...", "elapsed": 12.3}
        若模型已加载则直接返回: {"status": "already_loaded", "model_version": "..."}

    POST /model/unload
        无需请求体
        响应 (JSON): {"status": "unloaded"}
        卸载模型并释放 GPU 显存；若模型未加载则返回: {"status": "not_loaded"}
"""

import gc
import os
import sys
import uuid
import time
import hashlib
import logging
import argparse
import tempfile
import threading
import warnings

# 屏蔽第三方库的 FutureWarning 和 UserWarning，保持日志整洁
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import tornado.ioloop
import tornado.web
import tornado.httpserver

# 将项目根目录加入模块搜索路径，确保能导入 indextts 包
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 配置全局日志格式：时间 + 级别 + 消息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 全局变量 – 在 main() 中完成初始化后使用
# ---------------------------------------------------------------------------
tts_model = None                          # IndexTTS2 模型实例，通过接口动态加载/卸载
model_lock = threading.Lock()            # 保护模型加载/卸载的线程锁
default_spk_audio = "examples/voice_01.wav"  # 默认参考音频路径，可通过命令行参数覆盖

# 上传文件缓存目录（按文件 MD5 哈希存储，避免重复上传）
UPLOAD_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_files")

# 启动时的默认模型参数（由命令行参数填充），供 /model/load 使用
default_model_args = {
    "model_dir": "checkpoints",
    "cfg_path": "checkpoints/config.yaml",
    "fp16": True,
    "cuda_kernel": False,
    "deepspeed": False,
}


# ---------------------------------------------------------------------------
# 模型管理辅助函数
# ---------------------------------------------------------------------------

def _cleanup_upload_files():
    """
    清理 upload_files 目录中的所有缓存音频文件。
    在模型卸载时调用，释放磁盘空间。
    """
    if not os.path.isdir(UPLOAD_FILES_DIR):
        return
    removed = 0
    for fname in os.listdir(UPLOAD_FILES_DIR):
        fpath = os.path.join(UPLOAD_FILES_DIR, fname)
        try:
            os.remove(fpath)
            removed += 1
        except OSError as e:
            logger.warning("清理上传文件失败: %s — %s", fpath, e)
    if removed:
        logger.info("已清理 upload_files 目录，共删除 %d 个文件。", removed)


def _do_load_model(model_dir, cfg_path, fp16, cuda_kernel, deepspeed):
    """
    实际执行模型加载，返回 IndexTTS2 实例。
    调用方必须已持有 model_lock。
    """
    from indextts.infer_v2 import IndexTTS2
    logger.info("正在加载 IndexTTS2 模型，目录: '%s' ...", model_dir)
    model = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=fp16,
        use_cuda_kernel=cuda_kernel,
        use_deepspeed=deepspeed,
    )
    logger.info("模型加载完成。")
    return model


def _do_unload_model():
    """
    卸载全局模型实例，释放 GPU 显存。
    调用方必须已持有 model_lock。
    """
    global tts_model
    if tts_model is None:
        return False
    logger.info("正在卸载模型，释放显存...")
    # 逐一删除子模块引用，触发 Python 垃圾回收
    for attr in ("gpt", "semantic_model", "semantic_codec", "s2mel",
                 "campplus_model", "bigvgan", "qwen_emo"):
        if hasattr(tts_model, attr):
            delattr(tts_model, attr)
    tts_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU 显存已释放。")
    _cleanup_upload_files()
    logger.info("模型已卸载。")
    return True


# ---------------------------------------------------------------------------
# 请求处理器
# ---------------------------------------------------------------------------

class HealthHandler(tornado.web.RequestHandler):
    """
    健康检查接口处理器
    GET /health
    返回服务运行状态和模型版本信息，可用于 K8s/Docker 存活探针。
    """

    def get(self):
        version = getattr(tts_model, "model_version", None) if tts_model else None
        self.set_header("Content-Type", "application/json")
        self.finish({
            "status": "ok",
            "model_loaded": tts_model is not None,
            "model_version": version,
        })


class ModelStatusHandler(tornado.web.RequestHandler):
    """
    GET /model/status
    返回模型加载状态、版本号和设备信息。
    """

    def get(self):
        loaded = tts_model is not None
        self.set_header("Content-Type", "application/json")
        self.finish({
            "loaded": loaded,
            "model_version": getattr(tts_model, "model_version", None) if loaded else None,
            "device": getattr(tts_model, "device", None) if loaded else None,
        })


class ModelLoadHandler(tornado.web.RequestHandler):
    """
    POST /model/load
    动态加载模型，支持通过请求体覆盖默认参数。
    若模型已加载则直接返回当前状态，无需重复加载。
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self):
        self.set_status(204)
        self.finish()

    def post(self):
        import json as _json

        global tts_model

        # 若已加载，直接返回
        if tts_model is not None:
            self.finish({
                "status": "already_loaded",
                "model_version": getattr(tts_model, "model_version", None),
                "device": getattr(tts_model, "device", None),
            })
            return

        # 解析请求体（可选，覆盖默认参数）
        params = {}
        body = self.request.body
        if body:
            try:
                params = _json.loads(body.decode("utf-8"))
            except Exception as e:
                self.set_status(400)
                self.finish({"error": f"Invalid JSON: {e}"})
                return

        model_dir   = params.get("model_dir",   default_model_args["model_dir"])
        cfg_path    = params.get("cfg_path",    default_model_args["cfg_path"])
        fp16        = bool(params.get("fp16",        default_model_args["fp16"]))
        cuda_kernel = bool(params.get("cuda_kernel", default_model_args["cuda_kernel"]))
        deepspeed   = bool(params.get("deepspeed",   default_model_args["deepspeed"]))

        # 校验模型目录
        if not os.path.exists(model_dir):
            self.set_status(400)
            self.finish({"error": f"模型目录不存在: {model_dir}"})
            return
        for required_file in ("bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"):
            fp = os.path.join(model_dir, required_file)
            if not os.path.exists(fp):
                self.set_status(400)
                self.finish({"error": f"缺少必要模型文件: {fp}"})
                return

        # 用锁保证同一时刻只有一个请求执行加载
        if not model_lock.acquire(blocking=False):
            self.set_status(503)
            self.finish({"error": "模型正在加载/卸载中，请稍后重试"})
            return

        try:
            # 双重检查（锁内）
            if tts_model is not None:
                self.finish({
                    "status": "already_loaded",
                    "model_version": getattr(tts_model, "model_version", None),
                    "device": getattr(tts_model, "device", None),
                })
                return

            t0 = time.perf_counter()
            tts_model = _do_load_model(model_dir, cfg_path, fp16, cuda_kernel, deepspeed)
            elapsed = time.perf_counter() - t0

            self.finish({
                "status": "loaded",
                "model_version": getattr(tts_model, "model_version", None),
                "device": getattr(tts_model, "device", None),
                "elapsed": round(elapsed, 2),
            })
        except Exception as e:
            logger.exception("模型加载失败")
            tts_model = None
            self.set_status(500)
            self.finish({"error": str(e)})
        finally:
            model_lock.release()


class ModelUnloadHandler(tornado.web.RequestHandler):
    """
    POST /model/unload
    动态卸载模型并释放 GPU 显存。
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self):
        self.set_status(204)
        self.finish()

    def post(self):
        global tts_model

        if tts_model is None:
            self.finish({"status": "not_loaded"})
            return

        if not model_lock.acquire(blocking=False):
            self.set_status(503)
            self.finish({"error": "模型正在加载/卸载中，请稍后重试"})
            return

        try:
            unloaded = _do_unload_model()
            self.finish({"status": "unloaded" if unloaded else "not_loaded"})
        except Exception as e:
            logger.exception("模型卸载失败")
            self.set_status(500)
            self.finish({"error": str(e)})
        finally:
            model_lock.release()


class FileCheckHandler(tornado.web.RequestHandler):
    """
    GET /files/check?hash=<md5>
    检查指定 MD5 哈希的文件是否已存在于 upload_files 缓存目录。
    响应示例:
        {"exists": true,  "server_path": "upload_files/abc123.wav"}
        {"exists": false}
    """

    def get(self):
        file_hash = self.get_argument("hash", "").strip().lower()
        if not file_hash:
            self.set_status(400)
            self.finish({"error": "Missing 'hash' query parameter"})
            return

        os.makedirs(UPLOAD_FILES_DIR, exist_ok=True)
        # 匹配 upload_files/<hash>.<ext> 的任意文件
        for fname in os.listdir(UPLOAD_FILES_DIR):
            name_part = os.path.splitext(fname)[0]
            if name_part == file_hash:
                server_path = os.path.join(UPLOAD_FILES_DIR, fname)
                self.finish({"exists": True, "server_path": server_path})
                return

        self.finish({"exists": False})


class FileUploadHandler(tornado.web.RequestHandler):
    """
    POST /files/upload
    将上传的音频文件按其 MD5 哈希保存到 upload_files 缓存目录。
    表单字段:
        file    上传的音频文件（multipart/form-data）
    响应示例:
        {"hash": "abc123", "server_path": "upload_files/abc123.wav"}
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self):
        self.set_status(204)
        self.finish()

    def post(self):
        files = self.request.files.get("file")
        if not files:
            self.set_status(400)
            self.finish({"error": "No file uploaded. Use field name 'file'."})
            return

        uploaded = files[0]
        body = uploaded["body"]
        original_filename = uploaded.get("filename", "audio.wav")
        _, ext = os.path.splitext(original_filename)
        ext = ext.lower() if ext else ".wav"

        # 计算 MD5 哈希
        file_hash = hashlib.md5(body).hexdigest()

        os.makedirs(UPLOAD_FILES_DIR, exist_ok=True)
        dest_path = os.path.join(UPLOAD_FILES_DIR, f"{file_hash}{ext}")

        if not os.path.exists(dest_path):
            with open(dest_path, "wb") as f:
                f.write(body)
            logger.info("已保存上传文件: %s (%d bytes)", dest_path, len(body))
        else:
            logger.info("文件已存在，跳过保存: %s", dest_path)

        self.finish({"hash": file_hash, "server_path": dest_path})


class TTSHandler(tornado.web.RequestHandler):
    """
    语音合成接口处理器
    POST /tts
    接收 JSON 请求体，调用 IndexTTS2 推理，将生成的 WAV 文件以二进制形式返回。
    若模型未加载，返回 HTTP 503。
    """

    def set_default_headers(self):
        """设置 CORS 跨域响应头，允许任意来源的前端直接调用本接口。"""
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self):
        self.set_status(204)
        self.finish()

    def post(self):
        """
        核心接口：接收合成参数 → 推理生成 WAV → 返回音频二进制。
        模型未加载时返回 503，提示调用者先调用 /model/load。
        """
        import json as _json

        # ------------------------------------------------------------------ #
        # 0. 检查模型是否已加载                                                #
        # ------------------------------------------------------------------ #
        if tts_model is None:
            self.set_status(503)
            self.finish({"error": "模型未加载，请先调用 POST /model/load 加载模型"})
            return

        # ------------------------------------------------------------------ #
        # 1. 根据 Content-Type 选择解析方式                                    #
        # ------------------------------------------------------------------ #
        content_type = self.request.headers.get("Content-Type", "")

        # 用于记录本次请求中创建的所有临时文件路径，最终统一清理
        temp_files_to_cleanup = []

        if "multipart/form-data" in content_type:
            # ── 方式一：multipart/form-data ─────────────────────────────── #
            params = {}

            for key, val_list in self.request.body_arguments.items():
                params[key] = val_list[0].decode("utf-8") if val_list else ""

            def save_uploaded_file(field_name):
                files = self.request.files.get(field_name)
                if not files:
                    return None
                uploaded = files[0]
                original_filename = uploaded.get("filename", "audio.wav")
                _, ext = os.path.splitext(original_filename)
                ext = ext.lower() if ext else ".wav"
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="upload_spk_")
                os.close(tmp_fd)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded["body"])
                logger.info("已保存上传音频 [%s] → %s (%d bytes)",
                            field_name, tmp_path, len(uploaded["body"]))
                temp_files_to_cleanup.append(tmp_path)
                return tmp_path

            spk_audio_uploaded = save_uploaded_file("spk_audio_prompt")
            emo_audio_uploaded = save_uploaded_file("emo_audio_prompt")

            spk_audio_prompt = (
                spk_audio_uploaded
                or params.get("spk_audio_prompt_path", "").strip()
                or default_spk_audio
            )

            emo_audio_prompt_raw = (
                emo_audio_uploaded
                or params.get("emo_audio_prompt_path", "").strip()
                or None
            )

            emo_vector_raw = params.get("emo_vector", None)
            try:
                emo_vector = _json.loads(emo_vector_raw) if emo_vector_raw else None
            except Exception:
                emo_vector = None

            def get_bool(key, default=False):
                val = params.get(key, "").strip().lower()
                if val in ("1", "true", "yes"):
                    return True
                if val in ("0", "false", "no"):
                    return False
                return default

            text                = params.get("text", "").strip()
            emo_alpha           = float(params.get("emo_alpha", 1.0))
            use_emo_text        = get_bool("use_emo_text", False)
            emo_text            = params.get("emo_text", None) or None
            use_random          = get_bool("use_random", False)
            interval_silence_ms = int(params.get("interval_silence", 200))
            verbose             = get_bool("verbose", False)
            max_text_tokens     = int(params.get("max_text_tokens_per_segment", 120))

            generation_kwargs = {}
            for key, cast in (("do_sample", lambda v: v.lower() in ("1", "true", "yes")),
                               ("top_p", float), ("top_k", int), ("temperature", float),
                               ("length_penalty", float), ("num_beams", int),
                               ("repetition_penalty", float), ("max_mel_tokens", int)):
                if key in params and params[key] != "":
                    try:
                        generation_kwargs[key] = cast(params[key])
                    except (ValueError, TypeError):
                        pass

        else:
            # ── 方式二：application/json ─────────────────────────────────── #
            try:
                body = self.request.body
                if not body:
                    self.set_status(400)
                    self.finish({"error": "Empty request body"})
                    return
                params = _json.loads(body.decode("utf-8"))
            except Exception as e:
                self.set_status(400)
                self.finish({"error": f"Invalid JSON: {e}"})
                return

            spk_audio_uploaded   = None
            emo_audio_uploaded   = None

            spk_audio_prompt     = params.get("spk_audio_prompt", default_spk_audio)
            emo_audio_prompt_raw = params.get("emo_audio_prompt", None)
            emo_vector           = params.get("emo_vector", None)

            text                = params.get("text", "").strip()
            emo_alpha           = float(params.get("emo_alpha", 1.0))
            use_emo_text        = bool(params.get("use_emo_text", False))
            emo_text            = params.get("emo_text", None)
            use_random          = bool(params.get("use_random", False))
            interval_silence_ms = int(params.get("interval_silence", 200))
            verbose             = bool(params.get("verbose", False))
            max_text_tokens     = int(params.get("max_text_tokens_per_segment", 120))

            generation_kwargs = {}
            for key in ("do_sample", "top_p", "top_k", "temperature",
                        "length_penalty", "num_beams", "repetition_penalty", "max_mel_tokens"):
                if key in params:
                    generation_kwargs[key] = params[key]

        # ------------------------------------------------------------------ #
        # 2. 校验必填参数：text                                                #
        # ------------------------------------------------------------------ #
        if not text:
            self.set_status(400)
            self.finish({"error": "Field 'text' is required and must not be empty"})
            return

        # ------------------------------------------------------------------ #
        # 3. 校验参考音频路径                                                   #
        # ------------------------------------------------------------------ #
        if not os.path.exists(spk_audio_prompt):
            self.set_status(400)
            self.finish({"error": f"Speaker audio file not found: {spk_audio_prompt}"})
            return

        # ------------------------------------------------------------------ #
        # 4. 校验情绪参考音频路径（若有）                                        #
        # ------------------------------------------------------------------ #
        emo_audio_prompt = emo_audio_prompt_raw or None
        if emo_audio_prompt and not os.path.exists(emo_audio_prompt):
            self.set_status(400)
            self.finish({"error": f"Emotion audio file not found: {emo_audio_prompt}"})
            return

        # ------------------------------------------------------------------ #
        # 5. 调用模型推理                                                       #
        # ------------------------------------------------------------------ #
        tmp_output_path = None
        try:
            tmp_fd, tmp_output_path = tempfile.mkstemp(suffix=".wav", prefix="tts_out_")
            os.close(tmp_fd)

            logger.info("TTS request | text_len=%d | spk=%s | uploaded=%s",
                        len(text), spk_audio_prompt, spk_audio_uploaded is not None)
            t0 = time.perf_counter()

            tts_model.infer(
                spk_audio_prompt=spk_audio_prompt,
                text=text,
                output_path=tmp_output_path,
                emo_audio_prompt=emo_audio_prompt,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                emo_text=emo_text,
                use_random=use_random,
                interval_silence=interval_silence_ms,
                verbose=verbose,
                max_text_tokens_per_segment=max_text_tokens,
                **generation_kwargs,
            )

            elapsed = time.perf_counter() - t0
            logger.info("TTS done in %.2fs", elapsed)

            with open(tmp_output_path, "rb") as f:
                wav_bytes = f.read()

            out_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"

            self.set_header("Content-Type", "audio/wav")
            self.set_header("Content-Disposition", f'attachment; filename="{out_filename}"')
            self.set_header("Content-Length", str(len(wav_bytes)))
            self.set_header("X-Inference-Time", f"{elapsed:.3f}s")
            self.finish(wav_bytes)

        except Exception as e:
            logger.exception("TTS inference failed")
            self.set_status(500)
            self.finish({"error": str(e)})
        finally:
            if tmp_output_path and os.path.exists(tmp_output_path):
                try:
                    os.remove(tmp_output_path)
                except OSError:
                    pass
            for tmp in temp_files_to_cleanup:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                        logger.debug("已清理上传临时文件: %s", tmp)
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Tornado 应用工厂
# ---------------------------------------------------------------------------

def make_app():
    """
    创建并返回 Tornado Application 实例。
    路由表:
        /tts            → TTSHandler          语音合成接口
        /health         → HealthHandler       健康检查接口
        /model/status   → ModelStatusHandler  模型状态查询
        /model/load     → ModelLoadHandler    模型加载接口
        /model/unload   → ModelUnloadHandler  模型卸载接口
    """
    return tornado.web.Application(
        [
            (r"/tts",            TTSHandler),
            (r"/health",         HealthHandler),
            (r"/model/status",   ModelStatusHandler),
            (r"/model/load",     ModelLoadHandler),
            (r"/model/unload",   ModelUnloadHandler),
            (r"/files/check",    FileCheckHandler),
            (r"/files/upload",   FileUploadHandler),
        ],
        debug=False,
    )


# ---------------------------------------------------------------------------
# 程序入口
# ---------------------------------------------------------------------------

def main():
    global tts_model, default_spk_audio, default_model_args

    # ---------------------------------------------------------------------- #
    # 命令行参数定义                                                            #
    # ---------------------------------------------------------------------- #
    parser = argparse.ArgumentParser(
        description="IndexTTS2 Tornado HTTP Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",        type=str,  default="0.0.0.0",
                        help="服务监听地址，0.0.0.0 表示接受所有网卡")
    parser.add_argument("--port",        type=int,  default=8000,
                        help="服务监听端口")
    parser.add_argument("--model_dir",   type=str,  default="checkpoints",
                        help="模型权重文件目录")
    parser.add_argument("--cfg_path",    type=str,  default="checkpoints/config.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--fp16",        action="store_true", default=True,
                        help="启用 FP16 半精度推理（需要 GPU 支持）")
    parser.add_argument("--cuda_kernel", action="store_true", default=False,
                        help="启用 BigVGAN 自定义 CUDA 核加速")
    parser.add_argument("--deepspeed",   action="store_true", default=False,
                        help="启用 DeepSpeed 推理加速")
    parser.add_argument("--default_spk", type=str,  default="examples/voice_01.wav",
                        help="默认参考音频路径（请求中未指定时使用）")
    parser.add_argument("--auto_load",   action="store_true", default=False,
                        help="启动时自动加载模型（默认不加载，通过 POST /model/load 按需加载）")
    args = parser.parse_args()

    # ---------------------------------------------------------------------- #
    # 更新全局默认参数                                                          #
    # ---------------------------------------------------------------------- #
    default_model_args.update({
        "model_dir":   args.model_dir,
        "cfg_path":    args.cfg_path,
        "fp16":        args.fp16,
        "cuda_kernel": args.cuda_kernel,
        "deepspeed":   args.deepspeed,
    })

    if not os.path.exists(args.default_spk):
        logger.warning("默认参考音频文件不存在: %s", args.default_spk)
    default_spk_audio = args.default_spk

    # ---------------------------------------------------------------------- #
    # 可选：启动时自动加载模型                                                   #
    # ---------------------------------------------------------------------- #
    if args.auto_load:
        if not os.path.exists(args.model_dir):
            logger.error("模型目录 '%s' 不存在，请先下载模型文件。", args.model_dir)
            sys.exit(1)
        for required_file in ("bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"):
            fp = os.path.join(args.model_dir, required_file)
            if not os.path.exists(fp):
                logger.error("缺少必要模型文件: %s", fp)
                sys.exit(1)
        tts_model = _do_load_model(
            args.model_dir, args.cfg_path, args.fp16, args.cuda_kernel, args.deepspeed
        )
    else:
        logger.info("模型未自动加载。请通过 POST /model/load 接口按需加载模型。")

    # ---------------------------------------------------------------------- #
    # 启动 Tornado HTTP 服务                                                   #
    # ---------------------------------------------------------------------- #
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(args.port, address=args.host)
    logger.info("IndexTTS2 HTTP 服务已启动，监听地址: http://%s:%d", args.host, args.port)
    logger.info("可用接口:  POST /tts   GET /health   GET /model/status   POST /model/load   POST /model/unload   GET /files/check   POST /files/upload")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("收到退出信号，正在关闭服务...")
        tornado.ioloop.IOLoop.current().stop()


if __name__ == "__main__":
    main()

