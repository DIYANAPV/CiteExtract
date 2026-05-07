
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
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_safe_external_url(url: str) -> bool:
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

    try:
        ipaddress.ip_address(hostname)
        return not _is_unsafe_ip(hostname)
    except ValueError:
        pass

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
