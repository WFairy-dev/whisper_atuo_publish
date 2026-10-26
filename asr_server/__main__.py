"""启动：python -m asr_server"""
from __future__ import annotations

import uvicorn

from . import config


def main() -> None:
    uvicorn.run(
        "asr_server.app:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,  # 模型与队列在进程内，必须单 worker
        log_level="info",
    )


if __name__ == "__main__":
    main()
