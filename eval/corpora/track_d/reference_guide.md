# Microservice Framework Core Capabilities and Supported Modules

## 1. Supported Distributed Observability Protocols
The framework provides native OpenTelemetry integration for distributed request tracing. Traces are exported via OTLP gRPC over port 4317. Metrics are scraped via Prometheus HTTP endpoints at /metrics. W3C TraceContext headers (traceparent and tracestate) are propagated automatically across service boundaries.

## 2. Supported Caching Backends and Storage Policies
Redis version 6.2 and 7.0 are supported for distributed caching using redis-py client clustering. In-memory LRU caching is supported via local memory pools up to 2 GB per worker node. Memcached is explicitly not supported. Cache invalidation utilizes pub/sub channel messaging.

## 3. Supported Authentication Mechanisms
Authentication supports JSON Web Tokens (JWT) signed with RS256 or ES256 asymmetric cryptographic algorithms. Token expiration is enforced at 60 minutes. Basic authentication and session cookies are supported for legacy client integrations. OAuth2 authorization code grant flow is supported for external third-party logins.

## 4. Message Queuing Subsystem and Worker Architecture
The worker service uses PostgreSQL durable table-backed queues for reliable job execution with transactional guarantees. Poison message dead-letter routing triggers after 5 consecutive execution attempts. Amazon SQS and RabbitMQ are not natively supported in the core framework.
