# 语音转写远程服务（Whisper）

在 **AutoDL（GPU）** 上部署 Whisper 语音转写服务，供 **公司内网 CPU 机器** 上的业务流水线通过 HTTP 调用。

当前版本：**1.6.0**

## 架构

```text
公司内网 CPU（业务流水线）
        │  HTTP 上传音频 / 轮询结果（带 API Key）
        ▼
AutoDL 本服务（GPU 跑 Whisper turbo）
        │
        ├─ 上传音频：用完立即删除
        ├─ 转写结果：落盘保留 24 小时
        └─ 运行日志：按日期分文件夹，默认保留 3 天
```

- **不要**在内网 CPU 上跑 Whisper（会很卡）。
- 一次交来很多条时，服务先排队，再分给空闲的转写路。每一路单独一个进程、一份模型，一次只做一条，取消也只停这一路。路数写在 `asr_server/service.conf`，默认 2，本机最多按 3 路启动。

## 目录说明

| 路径 | 说明 |
|------|------|
| `asr_server/` | 服务端代码 |
| `asr_server/tmp/` | 临时上传（推理后清空） |
| `asr_server/results/` | 转写结果 JSON（保留 24 小时） |
| `asr_server/logs/` | 运行日志 |
| `client_example.py` | 内网侧调用示例（伪同步） |
| `scripts/start_asr.sh` | AutoDL 启动脚本 |
| `scripts/bench_asr.py` | 并发压测 |
| `docs/操作说明.md` | **完整操作手册（请先看这个）** |

## 快速开始（AutoDL）

```bash
# 1. 进入项目
cd /root/autodl-tmp/video_cut_forigen

# 2. 设置密钥（上线必改）
export ASR_API_KEY='请换成足够长的随机串'

# 3. 启动服务（默认端口 6006）
bash scripts/start_asr.sh
```

健康检查（无需密钥）：

```bash
curl http://127.0.0.1:6006/health
```

## 快速调用（公司内网）

```bash
export ASR_BASE_URL='http://<AutoDL可达地址>:6006'
export ASR_API_KEY='与服务器相同的密钥'

# 推荐：提交 + 轮询（流水线可当同步用）
python client_example.py /path/to/audio.mp4 --url "$ASR_BASE_URL" --api-key "$ASR_API_KEY"
```

代码里伪同步：

```python
from client_example import transcribe_wait

result = transcribe_wait(
    r"D:\wjl\视频\国外测试视频\剪辑测试v4\airport_retracked_redacted.mp4",
    base_url="http://<AutoDL可达地址>:6006",
    task_id="req_20261024_003",
    api_key="与服务器相同的密钥",
)
text = result["result"]["text"]
segments = result["result"]["segments"]
```

## 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（无需 Key） |
| `POST` | `/v1/asr` | 上传文件，立刻返回 `task_id` |
| `GET` | `/v1/asr/{task_id}` | 查询状态 / 取结果 |
| `POST` | `/v1/asr/{task_id}/cancel` | 立刻取消该任务 |
| `POST` | `/v1/asr/sync` | 上传并阻塞到完成（短任务调试用） |

鉴权请求头：`X-API-Key: <密钥>`

对接请看 **[docs/接口文档.md](docs/接口文档.md)**（基本信息、请求参数、请求示例、同步/异步响应、错误码）。

更细的步骤、环境变量、排障与压测见 **[docs/操作说明.md](docs/操作说明.md)**。
