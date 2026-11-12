"""启动：python -m dover_server"""
from __future__ import annotations

import uvicorn

from . import config


def main() -> None:
    uvicorn.run(
        "dover_server.app:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
