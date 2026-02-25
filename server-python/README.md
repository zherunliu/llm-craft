```bash
uv sync
source .venv/bin/activate
uv run uvicorn src.main:app --host 0.0.0.0 --port 3000 --reload
```

- 直接导入从 `sys.path` 中查找路径
- 相对导入从当前模块所在的路径（`__package__`）查找路径
- 当前运行的文件其`__package__`为 None，无法使用相对导入，可以使用 `-m` 参数指定模块路径来运行文件
