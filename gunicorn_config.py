"""
Gunicorn Configuration File
Optimized for Render.com deployment with memory constraints
"""
import multiprocessing
import os

# Server Socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker Processes
# Use only 1 worker to minimize memory usage on free tier
workers = 1
worker_class = "sync"
worker_connections = 1000
max_requests = 100  # Restart workers after N requests to prevent memory leaks
max_requests_jitter = 20  # Add randomness to prevent all workers restarting at once

# Timeouts
timeout = 120  # Worker timeout (seconds)
graceful_timeout = 30  # Time to wait for workers to finish handling requests
keepalive = 5  # Keep-alive connections

# Memory Management
preload_app = False  # Don't preload to reduce startup memory
# Set to True if you want to share memory across workers (use only if you have >1 worker)

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process Naming
proc_name = "wordwave_api"

# Server Mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed in the future)
# keyfile = None
# certfile = None

def on_starting(server):
    """Called just before the master process is initialized."""
    print("🚀 Starting Gunicorn server...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    print("🔄 Reloading Gunicorn server...")

def when_ready(server):
    """Called just after the server is started."""
    print("✅ Gunicorn server is ready. Accepting connections.")

def on_exit(server):
    """Called just before exiting Gunicorn."""
    print("👋 Shutting down Gunicorn server...")

def worker_int(worker):
    """Called when a worker receives the INT or QUIT signal."""
    print(f"⚠️  Worker {worker.pid} received INT signal")

def worker_abort(worker):
    """Called when a worker is aborted."""
    print(f"❌ Worker {worker.pid} aborted")