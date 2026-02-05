#!/usr/bin/env python3
"""
RTSP server for Jetson CSI camera.

- Prefers NVIDIA HW H.264 encoder (nvv4l2h264enc) if available
- Falls back to x264enc (CPU) if NVIDIA encoder plugin is missing
- Exposes: rtsp://<JETSON_IP>:8554/camera
"""

import os
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib  # noqa: E402


def _print_env_debug() -> None:
    print("---- Env debug ----")
    print("GST_PLUGIN_PATH:", os.environ.get("GST_PLUGIN_PATH", ""))
    print("GST_PLUGIN_SYSTEM_PATH:", os.environ.get("GST_PLUGIN_SYSTEM_PATH", ""))
    print("-------------------")


def has_element(name: str) -> bool:
    """Return True if a GStreamer element factory exists."""
    return Gst.ElementFactory.find(name) is not None


def build_pipeline(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    bitrate_bps: int = 4_000_000,
) -> str:
    """
    Build a pipeline string for RTSP.

    Uses nvarguscamerasrc (CSI camera). Encoder selection:
      - nvv4l2h264enc (HW) if available
      - x264enc (CPU) otherwise
    """
    base = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv ! "
    )

    pay = "rtph264pay name=pay0 pt=96 config-interval=1"

    if has_element("nvv4l2h264enc"):
        # HW encode (best)
        enc = f"nvv4l2h264enc insert-sps-pps=true bitrate={bitrate_bps} ! "
        pipeline = base + enc + pay
        print("[pipeline] Using HW encoder: nvv4l2h264enc")
        return pipeline

    if has_element("omxh264enc"):
        # Older Jetson encoder (sometimes present on older stacks)
        enc = f"omxh264enc bitrate={bitrate_bps} ! "
        pipeline = base + enc + pay
        print("[pipeline] Using encoder: omxh264enc")
        return pipeline

    # CPU fallback: convert to system memory + I420 for x264enc
    if not has_element("x264enc"):
        raise RuntimeError(
            "No H.264 encoder found. Install one of: nvv4l2h264enc (Jetson), omxh264enc, or x264enc."
        )

    enc = (
        "video/x-raw, format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30 ! "
    )
    pipeline = base + enc + pay
    print("[pipeline] Using CPU encoder: x264enc (fallback)")
    return pipeline


class CameraFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, pipeline: str):
        super().__init__()
        self._pipeline = pipeline

    def do_create_element(self, url):
        return Gst.parse_launch(self._pipeline)


def main() -> None:
    Gst.init(None)
    _print_env_debug()

    pipeline = build_pipeline(width=1280, height=720, fps=30, bitrate_bps=4_000_000)
    print("RTSP pipeline:\n ", pipeline)

    server = GstRtspServer.RTSPServer()
    server.set_service("8554")

    factory = CameraFactory(pipeline)
    factory.set_shared(True)

    mounts = server.get_mount_points()
    mounts.add_factory("/camera", factory)

    server.attach(None)

    print("RTSP stream ready:")
    print("  rtsp://<JETSON_IP>:8554/camera")
    GLib.MainLoop().run()


if __name__ == "__main__":
    main()
