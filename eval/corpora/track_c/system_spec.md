# System Specification and Infrastructure Architecture

## 1. Service Level Agreements and Operational Tiers

The distributed RAG evaluation cluster operates across three distinct service tiers. Each tier defines strict availability targets, latency percentiles, error budgets, and human escalation response times.

| Service Tier | Uptime Target | Max P99 Latency | Error Budget | Escalation Window | Primary Contact |
| --- | --- | --- | --- | --- | --- |
| Tier-1 Gateway | 99.99% | 50ms | 0.01% | 5 minutes | SRE Oncall Alpha |
| Tier-2 Compute | 99.90% | 200ms | 0.10% | 15 minutes | Platform Core Team |
| Tier-3 Batch | 99.00% | 2000ms | 1.00% | 60 minutes | Pipeline Operations |

## 2. Infrastructure Resource Allocation & Hardware Quotas

All compute nodes are provisioned on bare metal instances with dedicated NUMA nodes to prevent cross-socket memory bus contention. Disk volumes for vector indexes utilize direct-attached NVMe storage formatted with XFS.

| Node Role | Minimum vCPU | Minimum RAM | Storage Medium | Volume Size | Backup Frequency |
| --- | --- | --- | --- | --- | --- |
| Ingestion Worker | 8 vCPU | 32 GB | NVMe SSD | 500 GB | Daily differential |
| Vector Indexer | 16 vCPU | 64 GB | NVMe SSD | 1000 GB | Hourly snapshot |
| Query Coordinator | 4 vCPU | 16 GB | Standard SSD | 200 GB | Weekly archive |

## 3. Operational System Error Codes

- ERR-1001: Connection timeout between gateway and vector indexer (threshold: 3000ms). Triggered when socket handshake stalls during network route failover.
- ERR-1002: Insufficient memory allocation during dense embedding batch. Occurs when batch token count exceeds 32,768 tokens in single forward pass.
- ERR-2001: Query coordinator queue saturation exceeding 500 queued requests. Automatically initiates temporary backpressure and HTTP 429 throttling.
- ERR-3001: Vector dimension mismatch on SQLite index attachment. Occurs when embedding vector length deviates from index declaration dimension of 768.

## 4. Network Firewall and Port Configurations

- Port 8000: API Gateway HTTP entrypoint for public authenticated client traffic.
- Port 5432: PostgreSQL durable queue and metadata store (internal private VPC only).
- Port 11434: Local Ollama daemon REST interface for vector embeddings and inference.
- Port 9090: Prometheus metric scraper endpoint for telemetry and circuit breaker observability.
