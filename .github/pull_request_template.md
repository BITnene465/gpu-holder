## 摘要

- 

## 验证

- [ ] `python -m ruff check --no-cache src tests`
- [ ] `python -m pytest -q`
- [ ] `PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests`
- [ ] 行为变化已写入 README、docs 或示例

## 安全

- [ ] 本变更不会杀死、挂起、renice 或以其他方式控制外部 GPU 进程。
- [ ] 外部 GPU 进程 metadata 仍然只是只读调度信号。
- [ ] 真实 CUDA smoke test 只在空闲 GPU 或我明确拥有的 GPU 上运行。
