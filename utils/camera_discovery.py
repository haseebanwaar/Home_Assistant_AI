#!/usr/bin/env python3
"""
onvif_discover_and_identify.py

Discover ONVIF devices via WS-Discovery and try to fetch GetDeviceInformation.

Requirements:
    pip install onvif-zeep ws-discovery requests

Usage: edit CREDENTIALS list (username/password pairs to try) and run.
"""

import re
import socket
import time
from urllib.parse import urlparse

from requests.auth import HTTPDigestAuth
from wsdiscovery import WSDiscovery
from onvif import ONVIFCamera, exceptions as onvif_exceptions
import requests

# === User config ===
# MAC you want to match (optional). If you have MAC, we'll match later via ARP if needed.
TARGET_MAC = None  # e.g. "50:0f:f5:70:f5:08" or None

# Credentials to try against discovered devices. Put the camera's username/password first.
CREDENTIALS = [
    ("admin", "dummy"),
    ("admin", "dummy"),
    ("admin", "dummy"),
    ("admin", "dummy"),
]

HTTP_ENDPOINTS = [
    "/ISAPI/Streaming/channels/101/picture",
    "/cgi-bin/snapshot.cgi",
    "/cgi-bin/snapshot.jpg",
    "/jpg/image.jpg",
    "/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr={user}&pwd={pwd}"
]

WS_DISCOVERY_TIMEOUT = 5
ONVIF_TIMEOUT = 5
HTTP_TIMEOUT = 3


def parse_xaddrs(xaddrs_text):
    if not xaddrs_text:
        return []
    parts = xaddrs_text.split()
    return [p for p in parts if p.startswith("http")]


def try_onvif_get_info(host, port, user, pwd):
    """Try ONVIF GetDeviceInformation"""
    try:
        cam = ONVIFCamera(host, port, user, pwd)
        dev = cam.create_devicemgmt_service()
        info = dev.GetDeviceInformation()
        return {
            "manufacturer": getattr(info, "Manufacturer", None),
            "model": getattr(info, "Model", None),
            "firmware": getattr(info, "FirmwareVersion", None),
            "serial": getattr(info, "SerialNumber", None),
        }
    except onvif_exceptions.ONVIFError as e:
        return {"error": f"ONVIF error: {e}"}
    except Exception as e:
        return {"error": f"Exception: {e}"}


def try_onvif_stream_uri(host, port, user, pwd):
    """Try to get RTSP stream URI using ONVIF Media service."""
    try:
        cam = ONVIFCamera(host, port, user, pwd)
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if profiles:
            token = profiles[0]._token
            uri_resp = media.GetStreamUri(
                {'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                 'ProfileToken': token}
            )
            uri = uri_resp.Uri
            # Add credentials if not already included
            if user and "@" not in uri:
                parsed = urlparse(uri)
                auth_netloc = f"{user}:{pwd}@{parsed.hostname}"
                if parsed.port:
                    auth_netloc += f":{parsed.port}"
                uri = parsed._replace(netloc=auth_netloc).geturl()
            return uri
    except Exception as e:
        return None


def guess_rtsp_url(ip, user, pwd):
    """Return a guessed RTSP URL based on common paths."""
    common_paths = [
        # "/Streaming/Channels/101",
        "/cam/realmonitor?channel=1&subtype=0",
        # "/h264/ch1/main/av_stream",
        # "/live",
        # "/live/ch00_0",
    ]
    for path in common_paths:
        url = f"rtsp://{user}:{pwd}@{ip}:554{path}"
        # lightweight probe
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((ip, 554))
            s.send(f"OPTIONS rtsp://{ip}{path} RTSP/1.0\r\nCSeq: 1\r\n\r\n".encode())
            data = s.recv(1024).decode(errors="ignore")
            if "RTSP" in data:
                return url
        except Exception:
            continue
    return None


def discover_onvif_and_identify():
    wsd = WSDiscovery()
    wsd.start()
    try:
        print("[*] Discovering ONVIF devices...")
        services = wsd.searchServices(timeout=WS_DISCOVERY_TIMEOUT)
        if not services:
            print("[-] No ONVIF devices discovered.")
            return []

        devices = []
        for s in services:
            xaddrs = s.getXAddrs()
            epr = s.getEPR()
            scopes = s.getScopes()
            xaddrs_list = parse_xaddrs(xaddrs if not isinstance(xaddrs, list) else ' '.join(xaddrs))
            host = None
            port = None
            for x in xaddrs_list:
                try:
                    parsed = urlparse(x)
                    host = parsed.hostname
                    port = parsed.port or 80
                    break
                except Exception:
                    continue

            device = {"epr": epr, "host": host, "port": port, "xaddrs": xaddrs_list}
            print(f"\n[FOUND] {host}")

            if not host:
                devices.append(device)
                continue

            for user, pwd in CREDENTIALS:
                info = try_onvif_get_info(host, port, user, pwd)
                if "manufacturer" in info and info.get("manufacturer"):
                    device["onvif_info"] = info
                    device["credentials"] = {"user": user, "password": pwd}
                    print(f"  [+] ONVIF Auth success: {user}/{pwd}")

                    # Try ONVIF stream URI
                    rtsp = try_onvif_stream_uri(host, port, user, pwd)
                    if not rtsp:
                        rtsp = guess_rtsp_url(host, user, pwd)
                    if rtsp:
                        print(f"  [RTSP] {rtsp}")
                        device["rtsp_url"] = rtsp
                    else:
                        print("  [!] RTSP stream not found")
                    break
            devices.append(device)
        return devices

    finally:
        wsd.stop()


if __name__ == "__main__":
    devices = discover_onvif_and_identify()
    print("\n=== SUMMARY ===")
    for d in devices:
        print(d)