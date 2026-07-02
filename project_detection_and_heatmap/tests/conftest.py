"""
pytest conftest: 将项目根目录加入 sys.path, 确保 utils/ 可导入.
Add project root to sys.path so utils/ is importable.
"""
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
