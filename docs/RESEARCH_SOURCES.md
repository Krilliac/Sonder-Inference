# Research Sources

Snapshot links collected for the architecture research. Prefer official project repositories/docs and original papers.

## Engines / runtimes

- llama.cpp — https://github.com/ggml-org/llama.cpp
- GGML — https://github.com/ggml-org/ggml
- vLLM — https://github.com/vllm-project/vllm
- vLLM docs — https://docs.vllm.ai/
- SGLang — https://github.com/sgl-project/sglang
- TensorRT-LLM — https://github.com/NVIDIA/TensorRT-LLM
- TGI — https://github.com/huggingface/text-generation-inference
- LMDeploy — https://github.com/InternLM/lmdeploy
- LightLLM — https://github.com/ModelTC/lightllm
- DeepSpeed-MII — https://github.com/deepspeedai/DeepSpeed-MII
- Aphrodite Engine — https://github.com/PygmalionAI/aphrodite-engine
- MLC LLM — https://github.com/mlc-ai/mlc-llm
- WebLLM — https://github.com/mlc-ai/web-llm
- ExLlamaV3 — https://github.com/turboderp-org/exllamav3
- ExLlamaV2 — https://github.com/turboderp-org/exllamav2
- KTransformers — https://github.com/kvcache-ai/ktransformers
- NInfer — https://github.com/Neroued/ninfer
- ik_llama.cpp — https://github.com/ikawrakow/ik_llama.cpp
- ONNX Runtime GenAI — https://github.com/microsoft/onnxruntime-genai
- ExecuTorch — https://github.com/pytorch/executorch
- MLX — https://github.com/ml-explore/mlx
- MLX-LM — https://github.com/ml-explore/mlx-lm
- PowerInfer — https://github.com/SJTU-IPADS/PowerInfer
- BitNet — https://github.com/microsoft/BitNet
- FastFlowLM — https://github.com/FastFlowLM/FastFlowLM
- Lemonade — https://github.com/lemonade-sdk/lemonade
- Modular — https://github.com/modular/modular
- LitGPT — https://github.com/Lightning-AI/litgpt
- Candle — https://github.com/huggingface/candle
- LocalAI — https://github.com/mudler/LocalAI
- OpenVINO — https://github.com/openvinotoolkit/openvino

## Serving / distributed

- Ollama — https://github.com/ollama/ollama
- Triton Inference Server — https://github.com/triton-inference-server/server
- Ray Serve LLM — https://docs.ray.io/en/latest/serve/llm/
- NVIDIA Dynamo — https://github.com/ai-dynamo/dynamo
- llm-d — https://github.com/llm-d/llm-d
- BentoML — https://github.com/bentoml/BentoML
- OpenLLM — https://github.com/bentoml/OpenLLM
- TabbyAPI — https://github.com/theroyallab/tabbyAPI
- llamafile — https://github.com/Mozilla-Ocho/llamafile

## KV/cache and kernels

- LMCache — https://github.com/LMCache/LMCache
- Mooncake — https://github.com/kvcache-ai/Mooncake
- FlashInfer — https://github.com/flashinfer-ai/flashinfer
- FlashAttention — https://github.com/Dao-AILab/flash-attention
- MInference — https://github.com/microsoft/MInference

## Papers

- Orca, OSDI 2022 — https://www.usenix.org/conference/osdi22/presentation/yu
- PagedAttention / Efficient Memory Management for LLM Serving — https://arxiv.org/abs/2309.06180
- SGLang — https://arxiv.org/abs/2312.07104
- Sarathi — https://arxiv.org/abs/2308.16369
- DistServe — https://arxiv.org/abs/2401.09670
- Splitwise — https://arxiv.org/abs/2311.18677
- FastServe — https://arxiv.org/abs/2305.05920
- vAttention — https://arxiv.org/abs/2405.04437
- FlexGen — https://arxiv.org/abs/2303.06865
- Punica — https://arxiv.org/abs/2310.18547
- S-LoRA — https://arxiv.org/abs/2311.03285
- DeepSpeed-FastGen — https://arxiv.org/abs/2401.08671
- Preble — https://arxiv.org/abs/2407.00023
- P/D-Serve — https://arxiv.org/abs/2408.08147
- INFERCEPT — https://arxiv.org/abs/2402.01869
- Mooncake — https://arxiv.org/abs/2407.00079

## Surveys / catalogs

Use these to discover additional systems, but validate entries against primary sources:

- Awesome LLM Inference Engine — https://github.com/sihyeong/Awesome-LLM-Inference-Engine
- LLM Inference Handbook — https://github.com/h9-tec/LLM-Inference-Handbook

## Maintenance rule

For any dependency proposal, record:
- exact repository
- commit/tag evaluated
- date evaluated
- license at that revision
- supported hardware/model matrix
- benchmark configuration
- known correctness limitations
