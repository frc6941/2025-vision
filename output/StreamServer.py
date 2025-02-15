import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from typing import Optional

import cv2
from PIL import Image

from config.config import ConfigStore


class StreamServer:
    """Interface for outputing camera frames."""

    def start(self, config_store: ConfigStore) -> None:
        """Starts the output stream."""
        raise NotImplementedError

    def set_frame(self, frame: cv2.Mat) -> None:
        """Sets the frame to serve."""
        raise NotImplementedError


class MjpegServer(StreamServer):
    def __init__(self) -> None:
        self._frame: Optional[cv2.Mat] = None
        self._has_frame: bool = False

    def _make_handler(self):
        instance_streaming = StreamingHandler
        instance_streaming.mjpeg = self
        return instance_streaming

    class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    def _run(self, port: int) -> None:
        # noinspection PyTypeChecker
        server = self.StreamingServer(("", port), self._make_handler())
        server.serve_forever()

    def start(self, config_store: ConfigStore) -> None:
        threading.Thread(target=self._run, daemon=True, args=(config_store.local_config.stream_port,)).start()

    def set_frame(self, frame: cv2.Mat) -> None:
        self._frame = frame.copy()
        self._has_frame = True

    @property
    def frame(self) -> cv2.Mat:
        return self._frame

    @property
    def has_frame(self) -> bool:
        return self._has_frame


class StreamingHandler(BaseHTTPRequestHandler):
    HTML = """
<html>
<head>
    <title>Northstar Debug</title>
    <style>
        body {
            background-color: black;
        }

        img {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            max-width: 100%;
            max-height: 100%;
        }
    </style>
</head>
<body>
    <img src="stream.mjpg" />
</body>
</html>
    """
    mjpeg: MjpegServer = None

    def do_GET(self):
        if self.path == "/":
            content = self.HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    if not self.mjpeg.has_frame:
                        time.sleep(0.1)
                    else:
                        # TODO: remote config
                        # pil_im = cv2.resize(Image.fromarray(self_mjpeg._frame), (
                        #     ConfigStore.remote_config.camera_resolution_height * 0.5,
                        #     ConfigStore.remote_config.camera_resolution_width * 0.5))
                        pil_im = Image.fromarray(self.mjpeg.frame)
                        stream = BytesIO()
                        pil_im.save(stream, format="JPEG")
                        frame_data = stream.getvalue()

                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_data)))
                        self.end_headers()
                        self.wfile.write(frame_data)
                        self.wfile.write(b"\r\n")
            except Exception as e:
                print("Removed streaming client %s: %s", self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()
