"""
Realistic benchmark scenarios for contextprune.
All test data is domain-appropriate (software dev, APIs, databases).
No lorem ipsum — actual text that resembles real agent workloads.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Scenario 1 — Agent with memory files (~60% overlap across 3 memory files)
# ---------------------------------------------------------------------------

SCENARIO1_SYSTEM = """\
You are a senior software engineer AI assistant helping with a large Python backend project.
You have deep knowledge of Python best practices, REST API design, database schema design,
and distributed systems architecture. Always provide production-quality code with proper
error handling, logging, and tests. When answering, consider performance implications,
maintainability, and security. Follow the project's coding conventions: snake_case for
variables and functions, PascalCase for classes, 100-character line limit, type hints
everywhere, and docstrings for all public APIs. The project uses FastAPI, SQLAlchemy,
PostgreSQL, Redis, and Celery. Always suggest migrations when changing database schemas.
Prefer async/await patterns for I/O-bound operations. Use dependency injection via
FastAPI's Depends() mechanism. All endpoints must have request validation via Pydantic models.
"""

MEMORY_FILE_1 = """\
## Project Context — Backend Architecture
The project is a SaaS platform for managing software deployments. Core services include:
user authentication via JWT tokens, project management CRUD, deployment pipeline tracking,
and real-time status updates via WebSockets. The codebase is organized into domain modules:
auth/, projects/, deployments/, notifications/. Each module has its own router, models,
schemas, and service layer. Database uses PostgreSQL with SQLAlchemy ORM. Redis handles
caching and Celery task queues. FastAPI serves the HTTP layer. Authentication uses JWT
tokens stored in Redis with configurable TTL. The platform supports multi-tenancy via
organization scoping on all database models. All database queries must include org_id
filtering to prevent data leakage between tenants. The project uses Python 3.11 and
targets Python 3.12 compatibility. Type checking is enforced via mypy in strict mode.
Testing uses pytest with pytest-asyncio for async tests and factory_boy for fixtures.
You are a senior software engineer AI assistant helping with a large Python backend project.
You have deep knowledge of Python best practices, REST API design, database schema design,
and distributed systems architecture. Always provide production-quality code with proper
error handling, logging, and tests. The project uses FastAPI, SQLAlchemy, PostgreSQL,
Redis, and Celery. Always suggest migrations when changing database schemas.
"""

MEMORY_FILE_2 = """\
## Deployment Module Details
The deployment pipeline tracks these states: PENDING, QUEUED, RUNNING, SUCCESS, FAILED,
CANCELLED, TIMED_OUT. State transitions are managed by Celery workers. Each deployment
has associated build logs stored in S3 with presigned URL access. Deployments belong to
projects which belong to organizations. The deployment model has: id, project_id, org_id,
status, environment (staging/production), git_ref, started_at, completed_at, created_by.
Celery tasks handle: build_image, push_to_registry, deploy_to_k8s, run_health_checks.
Failed deployments trigger automatic rollback if the project has rollback enabled.
The notifications module sends Slack and email alerts on state changes. WebSocket
subscriptions allow real-time deployment status streaming to the frontend. Database uses
PostgreSQL with SQLAlchemy ORM. Redis handles caching and Celery task queues. FastAPI
serves the HTTP layer. Authentication uses JWT tokens stored in Redis with configurable
TTL. The platform supports multi-tenancy via organization scoping on all database models.
All queries must include org_id filtering to prevent data leakage between tenants.
"""

MEMORY_FILE_3 = """\
## Testing and Quality Standards
All new features require unit tests with >80% coverage. Integration tests use a test
PostgreSQL database spun up via docker-compose. End-to-end tests use pytest-playwright.
Test fixtures use factory_boy for model creation and faker for realistic test data.
Async tests use pytest-asyncio with asyncio_mode=auto. Mocking uses unittest.mock and
pytest-mock. CI runs on GitHub Actions with matrix testing across Python 3.11 and 3.12.
Code quality tools: black for formatting, isort for imports, flake8 for linting, mypy
for type checking, bandit for security scanning. Pre-commit hooks enforce formatting.
Coverage reports are uploaded to Codecov. Performance tests use locust for load testing.
You are a senior software engineer AI assistant helping with a large Python backend project.
You have deep knowledge of Python best practices, REST API design, and distributed systems.
Always provide production-quality code with proper error handling, logging, and tests.
The project uses FastAPI, SQLAlchemy, PostgreSQL, Redis, and Celery. Type checking is
enforced via mypy in strict mode. Testing uses pytest with pytest-asyncio for async tests.
"""

SCENARIO1_MESSAGES = [
    {"role": "user", "content": "How do I add a new deployment status field to track the Kubernetes pod name?"},
    {"role": "assistant", "content": "To add a `pod_name` field to the deployment model, you'll need to: 1) Add the column to the SQLAlchemy model, 2) Create an Alembic migration, 3) Update the Pydantic schemas, 4) Update the Celery tasks that set deployment status. Here's the migration code..."},
    {"role": "user", "content": "Should pod_name be nullable? What if the deployment fails before a pod is created?"},
    {"role": "assistant", "content": "Yes, pod_name should be nullable (Optional[str] in Python). Deployments can fail at the build stage before any Kubernetes pod is ever created, so making it non-nullable would require special handling for those failure cases. Use `Column(String(255), nullable=True)` in SQLAlchemy."},
    {"role": "user", "content": "What index should I add for querying deployments by pod_name?"},
    {"role": "assistant", "content": "A partial index would be most efficient: `Index('ix_deployments_pod_name', Deployment.pod_name, postgresql_where=Deployment.pod_name.isnot(None))`. This avoids indexing the many NULL values and keeps the index small."},
    {"role": "user", "content": "How do I test the Celery task that updates pod_name?"},
    {"role": "assistant", "content": "Use pytest with unittest.mock to mock the Kubernetes client. Patch `k8s_client.get_pod_name()` to return a test value, then assert the deployment's pod_name is updated after the task runs. Use factory_boy to create test deployments."},
    {"role": "user", "content": "Can you write the complete migration file?"},
]

def get_scenario1() -> Tuple[str, List[Dict[str, Any]]]:
    """Agent with memory files — high overlap scenario."""
    # Combine memory files into a rich system prompt
    system = SCENARIO1_SYSTEM + "\n\n" + MEMORY_FILE_1 + "\n\n" + MEMORY_FILE_2 + "\n\n" + MEMORY_FILE_3
    return system, SCENARIO1_MESSAGES


# ---------------------------------------------------------------------------
# Scenario 2 — RAG context (~40% overlap across 10 doc chunks)
# ---------------------------------------------------------------------------

SCENARIO2_SYSTEM = """\
You are a database architecture consultant. Answer questions using only the provided
documentation chunks. Cite which chunk your answer comes from. Be precise and technical.
"""

RAG_CHUNKS = [
    """\
[Chunk 1: PostgreSQL Index Types]
PostgreSQL supports several index types: B-tree (default, for equality and range queries),
Hash (equality only, no range), GiST (generalized search tree, for geometric/full-text data),
SP-GiST (space-partitioned GiST), GIN (generalized inverted index, ideal for arrays and JSONB),
BRIN (block range index, for naturally ordered large tables). B-tree indexes are the most
commonly used and support <, <=, =, >=, > operators. For JSONB columns containing arrays
of values, GIN indexes provide the best performance for containment (@>) and existence (?)
operators. Creating an index: CREATE INDEX CONCURRENTLY idx_name ON table(column) for
non-blocking index creation on large tables. Partial indexes using WHERE clauses reduce
index size and improve performance when queries filter on a subset of rows.
""",
    """\
[Chunk 2: Query Optimization Fundamentals]
The PostgreSQL query planner uses statistics collected by ANALYZE to estimate row counts
and choose optimal execution plans. Key concepts: sequential scan (full table read),
index scan (using an index to find rows), index-only scan (all data from index, no heap
access), bitmap heap scan (multiple indexes combined). Use EXPLAIN ANALYZE to see actual
execution plans with timing. Enable track_io_timing for I/O statistics. The planner
considers: index selectivity, table statistics, join strategies (nested loop, hash join,
merge join), parallelism opportunities. For B-tree indexes, PostgreSQL supports <, <=, =,
>=, > operators and ORDER BY optimization. Index selectivity is crucial — an index on a
boolean column rarely helps, while an index on a high-cardinality UUID column is very
effective. Run VACUUM ANALYZE regularly to keep statistics fresh.
""",
    """\
[Chunk 3: JSONB Performance]
JSONB stores data in a decomposed binary format, making it faster to process than JSON.
Key operators: -> (get JSON object field), ->> (get as text), #> (get nested), @> (contains),
<@ (contained by), ? (key exists), ?| (any key exists), ?& (all keys exist). For JSONB
queries, GIN indexes provide optimal performance. Two GIN operator classes: jsonb_ops
(default, indexes all keys and values) and jsonb_path_ops (indexes only values, smaller
and faster for containment queries). Expression indexes allow indexing specific JSONB
paths: CREATE INDEX ON events((payload->>'event_type')). This is more efficient than a
full GIN index when you always filter on the same key. JSONB columns work well for
semi-structured data where schema flexibility is needed alongside strong query performance.
PostgreSQL supports several index types including B-tree, Hash, GiST, GIN, and BRIN.
""",
    """\
[Chunk 4: Connection Pooling]
PostgreSQL uses a process-per-connection model, making connection management critical for
performance. Each connection spawns a backend process consuming ~5-10MB of memory. For
production applications, always use a connection pooler. PgBouncer is the most popular:
session mode (connection held for session duration), transaction mode (released after
each transaction, most efficient), statement mode (released after each statement, limited
compatibility). Recommended setup: application → PgBouncer (transaction mode) → PostgreSQL.
Configure max_connections in postgresql.conf based on available memory. A common formula:
max_connections = (RAM in GB * 100). pgBouncer pool_size should be 10-25% of max_connections.
For Django/SQLAlchemy, set pool_pre_ping=True to handle connection drops. Monitor connection
counts via pg_stat_activity. Dead connections waste resources; set idle timeouts.
""",
    """\
[Chunk 5: Partitioning Strategies]
PostgreSQL table partitioning splits large tables into smaller physical pieces for improved
query performance and easier data management. Three partition types: range (partition by
value ranges, ideal for time-series), list (explicit value lists), hash (distribute rows
evenly across partitions). Declarative partitioning syntax: CREATE TABLE events PARTITION
BY RANGE (created_at). Partitions inherit parent table structure. Partition pruning allows
the planner to skip irrelevant partitions. For time-series data, monthly or weekly range
partitions enable efficient data retention: DROP TABLE events_2023_01 removes a month
instantly versus DELETE which logs every row. Attach/detach partitions without locking:
ALTER TABLE events ATTACH PARTITION events_2024_01. Use PostgreSQL 14+ for improved
partition-wise joins and aggregates. Always include the partition key in WHERE clauses.
""",
    """\
[Chunk 6: Replication and High Availability]
PostgreSQL streaming replication uses WAL (Write-Ahead Log) to replicate changes to
standby servers. Primary-standby setup: primary writes WAL, standbys replay it. Synchronous
replication (synchronous_commit = on) ensures no data loss but adds latency. Asynchronous
replication is faster but risks data loss during failover. For HA, use Patroni with etcd
or Consul for automatic failover. Patroni manages promotion of standby to primary, updating
DNS/load balancer records. Hot standby allows read queries on standbys. PgBouncer in front
of Patroni handles connection rerouting during failover. Physical replication copies at the
block level — very efficient but requires same major version. Logical replication copies at
the row level — allows replication to different versions or selective table replication.
""",
    """\
[Chunk 7: Vacuum and Maintenance]
PostgreSQL uses MVCC (Multi-Version Concurrency Control) — updates create new row versions,
old versions accumulate until VACUUM removes them. Regular VACUUM is critical to prevent
table bloat and transaction ID wraparound (XID wraparound, a catastrophic condition).
Autovacuum runs automatically based on table activity thresholds. For high-write tables,
tune autovacuum: autovacuum_vacuum_scale_factor=0.01, autovacuum_analyze_scale_factor=0.005.
pg_toast handles oversized column values (>~2KB) via out-of-line storage. VACUUM FULL
reclaims space but requires exclusive lock — avoid in production. pg_repack is a safer
alternative. Monitor bloat via pgstattuple or bloat queries. Track dead tuples in
pg_stat_user_tables. Run VACUUM ANALYZE after bulk loads to update statistics for the
query planner. PostgreSQL query planner uses statistics from ANALYZE to choose execution plans.
""",
    """\
[Chunk 8: Monitoring and Observability]
Key PostgreSQL monitoring views: pg_stat_activity (current connections and queries),
pg_stat_user_tables (table-level statistics including seq_scan, idx_scan, dead tuples),
pg_stat_user_indexes (index usage statistics), pg_locks (current lock information),
pg_stat_bgwriter (background writer and checkpoint statistics). Enable pg_stat_statements
extension for per-query execution statistics (calls, total_time, mean_time, rows).
Set log_min_duration_statement to log slow queries. Use pgBadger to analyze PostgreSQL
logs. Prometheus + postgres_exporter for metrics collection. Key metrics to alert on:
connection count near max_connections, replication lag exceeding SLA, long-running queries,
autovacuum not keeping up, bloat ratio exceeding threshold. Always use a connection pooler.
""",
    """\
[Chunk 9: Security Best Practices]
PostgreSQL security layers: authentication (pg_hba.conf controls who can connect), roles
and privileges (GRANT/REVOKE), row-level security (RLS for multi-tenancy), column-level
security. Use scram-sha-256 authentication (replaces deprecated md5). Principle of least
privilege: create dedicated roles per service with only required permissions. Enable SSL
for all connections (ssl=on in postgresql.conf). Store connection strings in environment
variables or secrets managers, never in code. Audit logging: log_connections, log_disconnections,
log_statement='ddl'. For multi-tenant SaaS, Row Level Security is powerful:
CREATE POLICY tenant_isolation ON events USING (org_id = current_setting('app.org_id')::uuid).
Use pgAudit extension for comprehensive audit logging. Encrypt sensitive columns with
pgcrypto. Regularly rotate credentials. Scan for SQL injection via parameterized queries.
""",
    """\
[Chunk 10: Schema Design Patterns]
Effective PostgreSQL schema design: use UUIDs (uuid-ossp or gen_random_uuid()) for
distributed-friendly primary keys. Always include created_at TIMESTAMPTZ and updated_at
TIMESTAMPTZ columns with triggers for auto-update. Use NOT NULL constraints aggressively.
Foreign keys enforce referential integrity but add write overhead — consider whether strict
enforcement vs. application-level integrity is appropriate. Soft deletes: add deleted_at
column and partial index: CREATE INDEX ON records(id) WHERE deleted_at IS NULL. JSONB for
truly flexible/unknown fields, but prefer explicit columns for frequently queried data.
Enum types for fixed value sets. Check constraints for business rules. Naming conventions:
tables plural_snake_case, columns singular_snake_case, indexes ix_table_column. Use schemas
to organize tables by domain. Connection pooling is critical: use PgBouncer in transaction mode.
""",
]

SCENARIO2_MESSAGES = [
    {"role": "user", "content": "What index type should I use for a JSONB column that stores event payloads, where I frequently filter by event_type key?"},
]

def get_scenario2() -> Tuple[str, List[Dict[str, Any]]]:
    """RAG context — 10 doc chunks with ~40% overlap."""
    system = SCENARIO2_SYSTEM
    messages = list(SCENARIO2_MESSAGES)
    # Build system with all chunks prepended
    chunks_text = "\n\n".join(RAG_CHUNKS)
    system = SCENARIO2_SYSTEM + "\n\n## Documentation Chunks\n\n" + chunks_text
    return system, messages


# ---------------------------------------------------------------------------
# Scenario 3 — Tool-heavy agent (20 tools, only 2 topics in conversation)
# ---------------------------------------------------------------------------

SCENARIO3_SYSTEM = """\
You are an AI assistant with access to a comprehensive suite of developer tools.
Use tools when appropriate to help the user accomplish their task. Always prefer
using a tool over guessing when real data is needed. When using multiple tools,
plan your approach before executing. Return structured, actionable results.
"""

TOOL_SCHEMAS_20 = [
    {"name": "web_search", "description": "Search the web for current information, documentation, or examples. Returns top 10 results with titles, URLs, and snippets.", "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "num_results": {"type": "integer", "description": "Number of results (1-10)", "default": 5}}, "required": ["query"]}},
    {"name": "code_execute", "description": "Execute Python code in an isolated sandbox. Returns stdout, stderr, and execution time. Safe for data processing and calculations.", "input_schema": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "timeout_seconds": {"type": "integer", "description": "Max execution time", "default": 30}}, "required": ["code"]}},
    {"name": "calendar_read", "description": "Read calendar events for a given date range. Returns events with title, start/end time, attendees, and location.", "input_schema": {"type": "object", "properties": {"start_date": {"type": "string", "description": "ISO 8601 date"}, "end_date": {"type": "string", "description": "ISO 8601 date"}, "calendar_id": {"type": "string", "description": "Calendar ID", "default": "primary"}}, "required": ["start_date", "end_date"]}},
    {"name": "calendar_create", "description": "Create a new calendar event with title, description, attendees, location, and time. Returns the created event ID.", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "start_time": {"type": "string", "description": "ISO 8601 datetime"}, "end_time": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"}, "location": {"type": "string"}}, "required": ["title", "start_time", "end_time"]}},
    {"name": "weather_current", "description": "Get current weather conditions for a city. Returns temperature, humidity, wind speed, and precipitation.", "input_schema": {"type": "object", "properties": {"city": {"type": "string", "description": "City name or coordinates"}, "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"}}, "required": ["city"]}},
    {"name": "weather_forecast", "description": "Get 7-day weather forecast for a location. Returns daily high/low temperatures, precipitation probability, and conditions.", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "default": 7, "maximum": 7}}, "required": ["city"]}},
    {"name": "db_query", "description": "Execute a read-only SQL query against the production database. Returns results as JSON array. Query must start with SELECT.", "input_schema": {"type": "object", "properties": {"sql": {"type": "string", "description": "SQL SELECT query"}, "max_rows": {"type": "integer", "default": 100, "maximum": 1000}, "database": {"type": "string", "default": "prod"}}, "required": ["sql"]}},
    {"name": "db_schema", "description": "Get the schema for database tables. Returns columns, types, constraints, and indexes.", "input_schema": {"type": "object", "properties": {"table_name": {"type": "string", "description": "Table name or pattern (supports wildcards)"}, "include_indexes": {"type": "boolean", "default": True}}, "required": ["table_name"]}},
    {"name": "email_send", "description": "Send an email to one or more recipients. Supports HTML and plain text, attachments, and scheduling.", "input_schema": {"type": "object", "properties": {"to": {"type": "array", "items": {"type": "string"}}, "subject": {"type": "string"}, "body": {"type": "string"}, "html": {"type": "boolean", "default": False}, "attachments": {"type": "array", "items": {"type": "string"}}}, "required": ["to", "subject", "body"]}},
    {"name": "email_read", "description": "Read emails from inbox with optional filtering. Returns sender, subject, date, and body preview.", "input_schema": {"type": "object", "properties": {"folder": {"type": "string", "default": "inbox"}, "limit": {"type": "integer", "default": 10}, "from_address": {"type": "string"}, "subject_contains": {"type": "string"}}, "required": []}},
    {"name": "file_read", "description": "Read the contents of a file. Supports text files, JSON, YAML, CSV, and code files.", "input_schema": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute file path"}, "encoding": {"type": "string", "default": "utf-8"}, "max_bytes": {"type": "integer", "default": 1048576}}, "required": ["path"]}},
    {"name": "file_write", "description": "Write content to a file, creating it if it doesn't exist. Returns success status and bytes written.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "string", "enum": ["overwrite", "append", "create_only"], "default": "overwrite"}}, "required": ["path", "content"]}},
    {"name": "file_list", "description": "List files in a directory with optional filtering by extension or pattern. Returns file names, sizes, and modification times.", "input_schema": {"type": "object", "properties": {"directory": {"type": "string"}, "pattern": {"type": "string"}, "recursive": {"type": "boolean", "default": False}, "include_hidden": {"type": "boolean", "default": False}}, "required": ["directory"]}},
    {"name": "http_request", "description": "Make HTTP requests to external APIs. Supports GET, POST, PUT, PATCH, DELETE. Returns status code, headers, and body.", "input_schema": {"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"], "default": "GET"}, "headers": {"type": "object"}, "body": {"type": "string"}, "timeout": {"type": "integer", "default": 30}}, "required": ["url"]}},
    {"name": "image_generate", "description": "Generate an image from a text prompt using AI. Returns a URL to the generated image. Supports various sizes and styles.", "input_schema": {"type": "object", "properties": {"prompt": {"type": "string", "description": "Detailed image description"}, "size": {"type": "string", "enum": ["256x256", "512x512", "1024x1024"], "default": "512x512"}, "style": {"type": "string", "enum": ["realistic", "cartoon", "abstract", "sketch"]}}, "required": ["prompt"]}},
    {"name": "image_analyze", "description": "Analyze an image and extract information: objects, text (OCR), colors, and metadata.", "input_schema": {"type": "object", "properties": {"image_url": {"type": "string"}, "extract": {"type": "array", "items": {"type": "string"}, "description": "What to extract: objects, text, colors, faces"}}, "required": ["image_url"]}},
    {"name": "translate", "description": "Translate text between languages. Supports 100+ languages. Returns translated text and detected source language.", "input_schema": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string", "description": "Target language code (e.g. es, fr, de, ja)"}, "source_language": {"type": "string", "description": "Source language code. Auto-detected if omitted."}}, "required": ["text", "target_language"]}},
    {"name": "git_log", "description": "Get git commit history for a repository. Returns commits with hash, author, date, and message.", "input_schema": {"type": "object", "properties": {"repo_path": {"type": "string"}, "limit": {"type": "integer", "default": 20}, "branch": {"type": "string", "default": "main"}, "author": {"type": "string"}}, "required": ["repo_path"]}},
    {"name": "git_diff", "description": "Get the diff for a commit or between two commits. Returns file changes with line-by-line additions and deletions.", "input_schema": {"type": "object", "properties": {"repo_path": {"type": "string"}, "commit": {"type": "string"}, "base_commit": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["repo_path"]}},
    {"name": "slack_send", "description": "Send a message to a Slack channel or user. Supports formatting, attachments, and scheduled messages.", "input_schema": {"type": "object", "properties": {"channel": {"type": "string", "description": "Channel name (#general) or user (@username)"}, "message": {"type": "string"}, "thread_ts": {"type": "string", "description": "Thread timestamp for replies"}}, "required": ["channel", "message"]}},
]

SCENARIO3_MESSAGES = [
    {"role": "user", "content": "Run this Python code and tell me if it has any bugs: def fibonacci(n): if n <= 0: return [] elif n == 1: return [0] result = [0, 1]; [result.append(result[-1] + result[-2]) for _ in range(n-2)]; return result"},
    {"role": "assistant", "content": "I'll execute that code to test it. The logic looks correct at a glance — let me verify the output for various inputs."},
    {"role": "user", "content": "Great. Now write a faster version using memoization and execute both to compare performance on fibonacci(35)."},
]

def get_scenario3() -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Tool-heavy agent — 20 tools, conversation only about code execution."""
    return SCENARIO3_SYSTEM, SCENARIO3_MESSAGES, TOOL_SCHEMAS_20


# ---------------------------------------------------------------------------
# Scenario 4 — Repetitive chat (system prompt restated every few turns)
# ---------------------------------------------------------------------------

REPEATED_SYSTEM_BLOCK = """\
You are a helpful AI assistant. Always be concise, accurate, and professional.
Do not make up information. If you don't know something, say so. Prioritize
user safety. Follow all applicable laws and ethical guidelines. Do not assist
with harmful, illegal, or deceptive activities. Respond in the same language
as the user. Always be helpful, harmless, and honest.
"""

SCENARIO4_MESSAGES = [
    {"role": "user", "content": "What's the difference between REST and GraphQL?"},
    {"role": "assistant", "content": "REST uses fixed endpoints (one URL per resource), while GraphQL has a single endpoint and lets clients specify exactly what data they need. REST is simpler and cacheable by default; GraphQL reduces over/under-fetching and is better for complex, related data."},
    {"role": "user", "content": REPEATED_SYSTEM_BLOCK + "\n\nWhen should I choose GraphQL over REST?"},
    {"role": "assistant", "content": "Choose GraphQL when: clients need different data shapes (mobile vs desktop), you have complex related data (user → posts → comments), or you want a single API for multiple consumers. Stick with REST for simple CRUD APIs, when HTTP caching is critical, or when your team lacks GraphQL expertise."},
    {"role": "user", "content": "What about gRPC? How does it compare?"},
    {"role": "assistant", "content": "gRPC uses Protocol Buffers (binary serialization) over HTTP/2 — much faster than JSON over HTTP/1.1. It's ideal for internal microservice communication where performance matters. REST/GraphQL are better for public APIs since browsers and clients can consume them easily without special tooling."},
    {"role": "user", "content": REPEATED_SYSTEM_BLOCK + "\n\nCan gRPC be used from a browser?"},
    {"role": "assistant", "content": "Not directly — browsers can't make raw HTTP/2 gRPC calls. grpc-web is a proxy solution that translates between browser HTTP requests and gRPC. For browser-facing APIs, REST or GraphQL is typically more practical."},
    {"role": "user", "content": "What is Protocol Buffers and how do I define a schema?"},
    {"role": "assistant", "content": "Protocol Buffers (protobuf) is Google's binary serialization format. You define schemas in .proto files: `message User { string name = 1; int32 age = 2; }`. The compiler generates typed client/server code in your language. Compared to JSON, protobuf is 3-10x smaller and faster to serialize."},
    {"role": "user", "content": REPEATED_SYSTEM_BLOCK + "\n\nAre there any downsides to using protobuf?"},
    {"role": "assistant", "content": "Yes: binary format isn't human-readable (harder to debug), requires schema evolution discipline (field numbers must not be reused), and adds tooling complexity. For internal services with performance requirements it's worth it; for public APIs the debuggability tradeoff usually isn't."},
    {"role": "user", "content": "What's the best way to version a REST API?"},
    {"role": "assistant", "content": "URL versioning (/api/v1/, /api/v2/) is the most common and explicit. Header versioning (Accept: application/vnd.api+json;version=2) is cleaner but less visible. Avoid versioning via query params. For breaking changes: version aggressively. For non-breaking additions: just add them to the existing version."},
    {"role": "user", "content": REPEATED_SYSTEM_BLOCK + "\n\nHow do I handle backwards compatibility when deprecating API endpoints?"},
    {"role": "assistant", "content": "Use a Sunset header (RFC 8594) with the deprecation date. Maintain old endpoints for at least one major version cycle. Send Deprecation warnings in response headers. Communicate via changelog, email, and developer portal. Monitor usage of deprecated endpoints so you know when it's safe to remove them."},
    {"role": "user", "content": "What's OpenAPI and should I use it?"},
]

def get_scenario4() -> Tuple[str, List[Dict[str, Any]]]:
    """Repetitive chat — system prompt restated in user messages."""
    return REPEATED_SYSTEM_BLOCK, SCENARIO4_MESSAGES


# ---------------------------------------------------------------------------
# Scenario 5 — Code agent (large codebase context + redundant guidelines)
# ---------------------------------------------------------------------------

CODEBASE_CONTEXT = """\
## Repository: payment-service
## Language: Python 3.11
## Framework: FastAPI + SQLAlchemy + Stripe

### File: payment_service/models.py
```python
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid
from sqlalchemy import Column, String, Numeric, DateTime, ForeignKey, Index, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .database import Base

class PaymentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"

class Customer(Base):
    __tablename__ = "customers"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stripe_customer_id = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    payments = relationship("Payment", back_populates="customer")
    __table_args__ = (Index("ix_customers_org_id", "org_id"), Index("ix_customers_email", "email"))

class Payment(Base):
    __tablename__ = "payments"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stripe_payment_intent_id = Column(String(255), unique=True, nullable=False)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.id"), nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="usd")
    status = Column(SAEnum(PaymentStatus), nullable=False, default=PaymentStatus.PENDING)
    description = Column(String(500))
    metadata_ = Column("metadata", String)  # JSON string
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    customer = relationship("Customer", back_populates="payments")
    refunds = relationship("Refund", back_populates="payment")
    __table_args__ = (
        Index("ix_payments_org_id_status", "org_id", "status"),
        Index("ix_payments_customer_id", "customer_id"),
        Index("ix_payments_created_at", "created_at"),
    )

class Refund(Base):
    __tablename__ = "refunds"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stripe_refund_id = Column(String(255), unique=True, nullable=False)
    payment_id = Column(UUID(as_uuid=True), ForeignKey("payments.id"), nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    reason = Column(String(100))
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    payment = relationship("Payment", back_populates="refunds")
```

### File: payment_service/stripe_client.py
```python
import stripe
from typing import Optional, Dict, Any
from decimal import Decimal
from .config import settings
import logging

logger = logging.getLogger(__name__)
stripe.api_key = settings.STRIPE_SECRET_KEY

async def create_payment_intent(
    amount: Decimal,
    currency: str,
    customer_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> stripe.PaymentIntent:
    try:
        intent = stripe.PaymentIntent.create(
            amount=int(amount * 100),  # Stripe expects cents
            currency=currency.lower(),
            customer=customer_id,
            metadata=metadata or {},
            automatic_payment_methods={"enabled": True},
        )
        logger.info(f"Created payment intent {intent.id} for customer {customer_id}")
        return intent
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating payment intent: {e}")
        raise

async def retrieve_payment_intent(payment_intent_id: str) -> stripe.PaymentIntent:
    try:
        return stripe.PaymentIntent.retrieve(payment_intent_id)
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error retrieving payment intent {payment_intent_id}: {e}")
        raise

async def create_refund(payment_intent_id: str, amount: Optional[Decimal] = None) -> stripe.Refund:
    try:
        params: Dict[str, Any] = {"payment_intent": payment_intent_id}
        if amount is not None:
            params["amount"] = int(amount * 100)
        refund = stripe.Refund.create(**params)
        logger.info(f"Created refund {refund.id} for payment intent {payment_intent_id}")
        return refund
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating refund: {e}")
        raise
```

### File: payment_service/api/payments.py
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID
import logging
from ..database import get_db
from ..models import Payment, PaymentStatus
from ..schemas import PaymentCreate, PaymentResponse, PaymentListResponse
from ..stripe_client import create_payment_intent, retrieve_payment_intent
from ..auth import get_current_org
from ..exceptions import PaymentNotFoundError, InsufficientFundsError

router = APIRouter(prefix="/payments", tags=["payments"])
logger = logging.getLogger(__name__)

@router.post("/", response_model=PaymentResponse, status_code=status.HTTP_201_CREATED)
async def create_payment(
    payment_data: PaymentCreate,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_current_org),
):
    customer = await get_customer(db, payment_data.customer_id, org_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    intent = await create_payment_intent(
        amount=payment_data.amount,
        currency=payment_data.currency,
        customer_id=customer.stripe_customer_id,
        metadata={"org_id": str(org_id)},
    )
    
    payment = Payment(
        stripe_payment_intent_id=intent.id,
        customer_id=customer.id,
        org_id=org_id,
        amount=payment_data.amount,
        currency=payment_data.currency,
        description=payment_data.description,
        status=PaymentStatus.PENDING,
    )
    db.add(payment)
    await db.commit()
    await db.refresh(payment)
    return payment
```

### File: payment_service/schemas.py
```python
from pydantic import BaseModel, Field, validator
from decimal import Decimal
from typing import Optional
from uuid import UUID
from datetime import datetime
from .models import PaymentStatus

class PaymentCreate(BaseModel):
    customer_id: UUID
    amount: Decimal = Field(gt=0, le=999999.99)
    currency: str = Field(default="usd", regex="^[a-z]{3}$")
    description: Optional[str] = Field(None, max_length=500)

    @validator("amount")
    def validate_amount(cls, v):
        if v.as_tuple().exponent < -2:
            raise ValueError("Amount cannot have more than 2 decimal places")
        return v

class PaymentResponse(BaseModel):
    id: UUID
    stripe_payment_intent_id: str
    customer_id: UUID
    amount: Decimal
    currency: str
    status: PaymentStatus
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
```

### File: payment_service/tests/test_payments.py
```python
import pytest
from httpx import AsyncClient
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock
import stripe
from ..main import app
from ..models import Payment, PaymentStatus
from .factories import PaymentFactory, CustomerFactory, OrganizationFactory

@pytest.mark.asyncio
async def test_create_payment_success():
    mock_intent = MagicMock(spec=stripe.PaymentIntent)
    mock_intent.id = "pi_test_123"
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        with patch("payment_service.stripe_client.stripe.PaymentIntent.create", return_value=mock_intent):
            response = await client.post(
                "/payments/",
                json={"customer_id": "...", "amount": "99.99", "currency": "usd"},
                headers={"Authorization": "Bearer test_token"},
            )
    
    assert response.status_code == 201
    data = response.json()
    assert data["stripe_payment_intent_id"] == "pi_test_123"
    assert data["status"] == "pending"
    assert Decimal(data["amount"]) == Decimal("99.99")
```

### File: payment_service/config.py
```python
from pydantic import BaseSettings, validator
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    STRIPE_SECRET_KEY: str
    STRIPE_WEBHOOK_SECRET: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    MAX_DB_CONNECTIONS: int = 20
    
    @validator("DATABASE_URL")
    def validate_db_url(cls, v):
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL")
        return v
    
    class Config:
        env_file = ".env"

settings = Settings()
```
"""

REDUNDANT_GUIDELINES_1 = """\
## Development Guidelines
Always write type hints for all function parameters and return values. Use async/await
for all database operations. Follow the existing code patterns — look at adjacent files
before writing new code. All new API endpoints must have corresponding unit tests.
Error handling should use custom exception classes from exceptions.py. Logging should
use the module-level logger: `logger = logging.getLogger(__name__)`. Never log sensitive
data like payment amounts, card numbers, or personal information. Always scope database
queries with org_id to prevent cross-tenant data access. Use Pydantic validators for
input validation. Follow PEP 8 style guidelines. Use snake_case for functions and
variables. Use PascalCase for class names. Keep functions under 50 lines.
"""

REDUNDANT_GUIDELINES_2 = """\
## Code Quality Requirements
Write type hints for all function parameters and return values. Use async/await patterns
for all I/O operations including database queries. Always look at nearby code before
writing new implementations to maintain consistency. New features require unit tests
before merge. Use the custom exception classes defined in exceptions.py for error
handling. Use the module logger: `logger = logging.getLogger(__name__)` at the top of
each file. Never include sensitive information in log messages — no payment amounts,
card details, or personal data in logs. Scope all database queries by org_id for
multi-tenant isolation. Validate all inputs using Pydantic models and validators.
Follow Python PEP 8 style guide. snake_case for functions/variables, PascalCase for
classes. Functions should stay under 50 lines for maintainability.
"""

SCENARIO5_MESSAGES = [
    {"role": "user", "content": "I need to add webhook handling for Stripe payment events. The webhook should update payment status when a payment_intent.succeeded or payment_intent.payment_failed event comes in. What's the best approach?"},
    {"role": "assistant", "content": "I'll design a webhook handler that integrates cleanly with your existing payment service architecture..."},
    {"role": "user", "content": "Can you write the complete webhook endpoint including signature verification, event processing, and the database update logic?"},
]

def get_scenario5() -> Tuple[str, List[Dict[str, Any]]]:
    """Code agent — large codebase context with redundant guidelines."""
    system = CODEBASE_CONTEXT + "\n\n" + REDUNDANT_GUIDELINES_1 + "\n\n" + REDUNDANT_GUIDELINES_2
    return system, SCENARIO5_MESSAGES


# ---------------------------------------------------------------------------
# Tool recall scenarios (Exp 3)
# ---------------------------------------------------------------------------

TOOL_POOL_RECALL = TOOL_SCHEMAS_20  # reuse the 20-tool pool

RECALL_TEST_CASES = [
    {
        "query": "Search for the latest Python documentation on async/await",
        "correct_tools": ["web_search"],
        "messages": [{"role": "user", "content": "Search for the latest Python documentation on async/await syntax and best practices"}],
    },
    {
        "query": "Execute a Python script to calculate fibonacci numbers",
        "correct_tools": ["code_execute"],
        "messages": [{"role": "user", "content": "Execute this Python code: def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2); print(fib(10))"}],
    },
    {
        "query": "Schedule a team meeting next Tuesday at 2pm",
        "correct_tools": ["calendar_create", "calendar_read"],
        "messages": [{"role": "user", "content": "Create a calendar event for a team meeting next Tuesday at 2pm with the backend team"}],
    },
    {
        "query": "What's the weather in San Francisco today?",
        "correct_tools": ["weather_current"],
        "messages": [{"role": "user", "content": "What is the current weather in San Francisco? Is it going to rain?"}],
    },
    {
        "query": "Query the database for all failed payments in the last 7 days",
        "correct_tools": ["db_query", "db_schema"],
        "messages": [{"role": "user", "content": "Run a database query to get all failed payments from the last 7 days with their customer IDs and amounts"}],
    },
    {
        "query": "Send an email notification to the team about the deployment",
        "correct_tools": ["email_send"],
        "messages": [{"role": "user", "content": "Send an email to the team at team@company.com saying the deployment to production succeeded"}],
    },
    {
        "query": "List all Python files in the src directory",
        "correct_tools": ["file_list", "file_read"],
        "messages": [{"role": "user", "content": "List all .py files in the /home/user/project/src directory and read the main.py file"}],
    },
    {
        "query": "Make a POST request to the Stripe API to create a customer",
        "correct_tools": ["http_request"],
        "messages": [{"role": "user", "content": "Make an HTTP POST request to https://api.stripe.com/v1/customers with name and email parameters"}],
    },
    {
        "query": "Generate an image of a Python snake for the documentation",
        "correct_tools": ["image_generate"],
        "messages": [{"role": "user", "content": "Generate an image of a cartoon Python snake wearing glasses for our API documentation header"}],
    },
    {
        "query": "Translate the error message from Japanese to English",
        "correct_tools": ["translate"],
        "messages": [{"role": "user", "content": "Translate this Japanese error message to English: エラーが発生しました。しばらく待ってから再試行してください。"}],
    },
]


def get_all_scenarios():
    """Return all scenarios as a list of (name, system, messages, tools) tuples."""
    s1_sys, s1_msgs = get_scenario1()
    s2_sys, s2_msgs = get_scenario2()
    s3_sys, s3_msgs, s3_tools = get_scenario3()
    s4_sys, s4_msgs = get_scenario4()
    s5_sys, s5_msgs = get_scenario5()
    return [
        ("Agent+Memory", s1_sys, s1_msgs, None),
        ("RAG Context", s2_sys, s2_msgs, None),
        ("Tool-Heavy", s3_sys, s3_msgs, s3_tools),
        ("Repetitive Chat", s4_sys, s4_msgs, None),
        ("Code Agent", s5_sys, s5_msgs, None),
    ]
