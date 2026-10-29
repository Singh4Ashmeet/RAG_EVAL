# Benchmarks
Benchmark results and methodology (placeholder).

## LLM Inference

<!-- BEGIN:llm -->

### Environment

- **Date**: 2026-10-29T14:04:14.146880+00:00
- **OS**: Windows 11 (10.0.26200)
- **CPU**: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- **Cores**: 8 physical / 16 logical
- **RAM**: 13.3 GB
- **Ollama**: 0.40.13
- **AC Power**: ✅ Plugged in
- **Prompt tokens**: 614

### Results

| Model       | Size   | Cold TTFT | Warm TTFT | Prompt tok/s | Gen tok/s | Peak Ollama RSS | Min Avail RAM |
| ----------- | ------ | --------- | --------- | ------------ | --------- | --------------- | ------------- |
| llama3.2:3b | 2.4 GB | 18.45 s   | 12.83 s   | 57.7         | 11.0      | 2.6 GB          | 1.4 GB        |

<!-- END:llm -->
