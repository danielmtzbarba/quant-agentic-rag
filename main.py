import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / "src").resolve()))

main = import_module("stock_agent_rag.cli").main


if __name__ == "__main__":
    main()
