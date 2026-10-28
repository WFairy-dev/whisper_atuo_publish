"""启动：python -m ocr_server"""
from __future__ import annotations

import uvicorn

from . import config


def main() -> None:
    uvicorn.run(
        "ocr_server.app:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,  # 对外仍是一个进程；识别路数由引擎另开子进程
        log_level="info",
    )


if __name__ == "__main__":
    main()
