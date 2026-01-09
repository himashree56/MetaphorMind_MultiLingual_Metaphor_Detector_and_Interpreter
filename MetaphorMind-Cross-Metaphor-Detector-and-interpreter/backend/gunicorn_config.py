# Gunicorn Configuration for Production Deployment
# This configuration enables Uvicorn workers with Gunicorn for better concurrency

import multiprocessing
import os

# ============================================================================
# SERVER SOCKET
# ============================================================================

# The socket to bind
bind = "0.0.0.0:8000"

# The number of worker processes for handling requests
# Formula: (2 x CPU cores) + 1
# For 4 cores: (2 x 4) + 1 = 9 workers
# Adjust based on your server's CPU cores
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# The type of workers to use
# uvicorn.workers.UvicornWorker provides async support for FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Worker connections (max concurrent connections per worker)
worker_connections = 1000

# ============================================================================
# TIMEOUT
# ============================================================================

# Workers silent for more than this many seconds are killed and restarted
# Set to 120 for ML model predictions (can take time)
timeout = 120

# Graceful timeout for worker shutdown
graceful_timeout = 30

# ============================================================================
# LOGGING
# ============================================================================

# The address to bind to for the Gunicorn status socket
# This is used for monitoring
bind_unix = None

# Log level
loglevel = "info"

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Access log file
accesslog = "-"  # Log to stdout

# Error log file
errorlog = "-"  # Log to stderr

# ============================================================================
# PROCESS NAMING
# ============================================================================

# A base to use with setproctitle for process naming
proc_name = "metaphor-detector-api"

# ============================================================================
# SERVER MECHANICS
# ============================================================================

# Detach the server from the controlling terminal
daemon = False

# The maximum number of pending connections
backlog = 2048

# The number of seconds to wait for requests on a Keep-Alive connection
keepalive = 5

# ============================================================================
# SSL
# ============================================================================

# SSL version to use (if needed)
# ssl_version = "TLSv1_2"

# Path to SSL certificate (if needed)
# certfile = "/path/to/certfile"

# Path to SSL key file (if needed)
# keyfile = "/path/to/keyfile"

# ============================================================================
# SECURITY
# ============================================================================

# Limit the allowed hosts
# limit_request_line = 4094
# limit_request_fields = 100
# limit_request_field_size = 8190

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# The maximum number of requests a worker will process before being restarted
max_requests = 1000

# The maximum jitter to add to the max_requests setting
max_requests_jitter = 50

# Preload application code before forking worker processes
preload_app = True

# ============================================================================
# DEVELOPMENT
# ============================================================================

# Reload workers when Python code changes (development only)
reload = False

# ============================================================================
# NOTES
# ============================================================================

"""
DEPLOYMENT GUIDE:

1. INSTALLATION:
   pip install gunicorn uvicorn[standard]

2. DEVELOPMENT (Single Worker - Fast Reload):
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

3. PRODUCTION (Multiple Workers - Better Concurrency):
   gunicorn -c gunicorn_config.py main:app

4. CUSTOM WORKER COUNT:
   GUNICORN_WORKERS=10 gunicorn -c gunicorn_config.py main:app

5. WITH NGINX REVERSE PROXY:
   - Run Gunicorn on localhost:8000
   - Configure Nginx to proxy requests to http://localhost:8000
   - Nginx handles SSL, load balancing, and static files

SCALING:
   - For 10-20 simultaneous users: 4-8 workers
   - For 50-100 simultaneous users: 8-16 workers
   - For 200+ simultaneous users: 16+ workers + load balancing

MONITORING:
   - Monitor CPU usage per worker
   - Monitor memory usage
   - Monitor request latency
   - Adjust worker count based on metrics

PERFORMANCE TIPS:
   - Use preload_app = True to share models across workers
   - Set timeout appropriately for ML predictions
   - Use caching to reduce model inference calls
   - Monitor and adjust max_requests based on memory leaks
"""
