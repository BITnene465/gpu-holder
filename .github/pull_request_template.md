## 变更摘要

- 

## 验证

- [ ] `python -m ruff check --no-cache src tests`
- [ ] `python -m pytest -q`
- [ ] `PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests`
- [ ] 行为变化已同步更新文档或示例

## 安全

- [ ] 本次变更不会 kill、suspend、renice 或以其他方式控制外部 GPU 进程。
- [ ] 外部 GPU 进程元数据只作为只读调度信号。
- [ ] 真实 CUDA smoke 只在空闲 GPU 或测试者明确拥有的 GPU 上运行。
