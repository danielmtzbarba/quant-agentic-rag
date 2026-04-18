import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-mock-key")
sys.path.insert(0, str((Path(__file__).parent / "src").resolve()))

from stock_agent_rag.workflow import build_app


def main() -> None:
    app = build_app()
    print("Graph nodes:", app.get_graph().nodes.keys())


if __name__ == "__main__":
    main()
