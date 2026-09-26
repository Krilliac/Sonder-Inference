# Ollama baseline: PENDING

Status as of 2026-09-26: **not captured.**

This branch was built in an isolated staging environment while the target
workstation (DESKTOP-950K3EI, Ollama on `127.0.0.1:11434`) was offline. No live
Ollama was reachable, so no numbers exist. No models were pulled and no
placeholder numbers were invented.

## Capture (keep the run under about 10 minutes, and don't compete with live Sonder traffic)

From the repository root on the workstation, after building (for example,
`scripts/build.ps1`, or `cmake --preset msvc-release` then
`cmake --build build/msvc-release -j 4`):

```powershell
# 1. Pick an installed model (no pulls)
.\build\msvc-release\bench\sonder-bench.exe --backend ollama --list-models

# 2. Small baseline. --require-idle refuses to run if a different model is resident.
.\build\msvc-release\bench\sonder-bench.exe --backend ollama --model <model> `
    --warmup 1 --runs 3 --budget-seconds 540 --require-idle `
    --hardware "<GPU + VRAM, CPU, RAM, driver/CUDA version>" --label ollama-baseline-v1
```

Commit the generated `bench/results/<date>-ollama-<model>-baseline.json` and
`.md`, update `README.md` in this directory, and delete this file.
