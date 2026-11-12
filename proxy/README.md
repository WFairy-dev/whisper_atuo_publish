# AutoDL 多服务代理

平台自定义服务主要映射实例内 **6006** 与 **6008**。要同时对外提供语音与清晰度时，在 6006 上跑官方轻量代理，按路径转发到本机各服务。

参考：[暴露多个服务](https://www.autodl.com/docs/proxy_in_instance/)

`route_path` 须写成 Gin 通配形式（例如 `/v1/asr` + `/v1/asr/*`），不要写成 `/v1/asr*`。

## 端口约定

| 对外入口 | 本机实际服务 | 路径 |
|---------|-------------|------|
| 平台 6006 映射 | `proxy_in_instance` | 见下 |
| → | 语音 `7006` | `/v1/asr*`（探活只认 `/v1/asr/health`） |
| → | 清晰度 `6009` | `/v1/vqa*`（含 `/v1/vqa/health`） |
| 平台 6008 映射 | 文字识别 `6008` | `/v1/ocr*`、`/health` |

## 启动顺序

1. `bash scripts/start_asr.sh`（默认已改为 7006）
2. `bash scripts/start_ocr.sh`（6008）
3. `bash scripts/start_dover.sh`（6009）
4. `bash scripts/start_proxy.sh`（6006）

停止代理：`kill "$(cat proxy/logs/supervise.pid)"`
