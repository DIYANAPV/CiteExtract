"""URL safety check for fetches initiated from user-supplied references.

Citation references can contain attacker-controlled URLs. Without a check,
the verifier becomes an SSRF oracle: a reference of the form
`http://169.254.169.254/latest/meta-data/iam/...` would be fetched and the
response status leaked back into the verification report.

`is_safe_external_url` rejects:
  - non-http(s) schemes (file://, gopher://, ftp://, etc.)
  - empty hostnames
  - hostnames that resolve to private, loopback, link-local, multicast,
    reserved, or unspecified IPs (covers RFC 1918, 127/8, 169.254/16,
    fc00::/7, fe80::/10, ::, etc.)
  - hostnames that fail to resolve at all (fail closed)

Hostname resolution uses `socket.getaddrinfo`, which is blocking. Async
callers should wrap the call in `asyncio.to_thread` if the latency matters.
The check is cheap in practice — a few ms cached by the OS resolver.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

log = logging.getLogger(__name__)


_ALLOWED_SCHEMES = {"http", "https"}


def _is_unsafe_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable — treat as unsafe
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_safe_external_url(url: str) -> bool:
    """Return True only if `url` is safe to fetch from an untrusted source.

    Fail-closed semantics: any error during parsing or DNS resolution
    returns False rather than letting an unverified URL through.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    # If the hostname is itself a literal IP, check it directly.
    try:
        ipaddress.ip_address(hostname)
        return not _is_unsafe_ip(hostname)
    except ValueError:
        pass  # hostname is a domain name — fall through to DNS resolution

    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False

    if not infos:
        return False

    for info in infos:
        ip_str = info[4][0]
        if _is_unsafe_ip(ip_str):
            return False

    return True
