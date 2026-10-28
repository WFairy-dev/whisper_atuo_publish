"""启动：python -m ocr_server"""
from __future__ import annotations

import uvicorn

from . import config


def main() -> None:
    uvicorn.run(
        "ocr_server.app:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,  # 模型只加载一份，必须单进程
        log_level="info",
    )


if __name__ == "__main__":
    main()
