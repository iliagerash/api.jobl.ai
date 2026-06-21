"""
serve.py — Launch the API server with mode selection.

Usage:
    python serve.py                                    # process mode, 1 worker
    python serve.py --mode process --workers 3         # process mode, 3 workers
    python serve.py --mode extractor                   # extractor mode, 1 worker
    python serve.py --mode extractor --port 8002       # extractor on different port
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Jobl API server")
    parser.add_argument("--mode", choices=["process", "extractor"], default="process")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    os.environ["SERVER_MODE"] = args.mode

    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
