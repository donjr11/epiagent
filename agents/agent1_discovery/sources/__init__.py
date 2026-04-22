"""Ingestion source clients.

Each source exposes a callable that yields a uniform record dict. All network
I/O goes through these modules so tests can monkey-patch a single seam.
"""
