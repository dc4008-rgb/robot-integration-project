"""Jetson 上的实时检测节点：浏览器/本地窗口显示 + ROS2 发布检测结果。

发布话题:
    /detections        vision_msgs/Detection2DArray   类别 id、置信度、检测框
    /detection_image   sensor_msgs/Image              画好框的图像（可在 rviz2 里看）

用法:
    python3 yolo_ros2_node.py --weights best.engine --source 0
    python3 yolo_ros2_node.py --weights best.pt --source 0 --no-display --web-port 8080
"""
import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255)]

VIEWER_HTML = b"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO Live</title>
<style>
:root{color-scheme:dark;--ink:#f4f3ee;--muted:#a6aaa7;--panel:#161918;--live:#45d483}
*{box-sizing:border-box}body{margin:0;background:#090b0a;color:var(--ink);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden}
header{height:48px;padding:0 18px;display:flex;align-items:center;justify-content:space-between;background:var(--panel);border-bottom:1px solid #2b302e}
.brand{font-weight:700}.status{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}.dot{width:8px;height:8px;border-radius:50%;background:var(--live);box-shadow:0 0 12px var(--live)}
main{height:calc(100vh - 48px);display:grid;place-items:center;background:repeating-linear-gradient(0deg,#0c0f0e 0,#0c0f0e 1px,#090b0a 1px,#090b0a 4px)}
img{display:block;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain}
</style>
</head>
<body><header><div class="brand">YOLO LIVE</div><div class="status"><span class="dot"></span><span>JETSON / ROS2</span></div></header><main><img src="/stream.mjpg" alt="Real-time detection"></main></body>
</html>"""


class MjpegServer:
    def __init__(self, port: int, jpeg_quality: int = 80) -> None:
        self.jpeg_quality = jpeg_quality
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.status = {"ready": False, "fps": 0.0, "objects": 0}

        stream = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    self._send_bytes("text/html; charset=utf-8", VIEWER_HTML)
                elif self.path == "/snapshot.jpg":
                    with stream.condition:
                        frame = stream.frame
                    if frame is None:
                        self.send_error(503, "No frame yet")
                    else:
                        self._send_bytes("image/jpeg", frame)
                elif self.path == "/status.json":
                    self._send_bytes(
                        "application/json",
                        json.dumps(stream.status).encode("utf-8"),
                    )
                elif self.path == "/stream.mjpg":
                    self._stream_frames()
                else:
                    self.send_error(404)

            def _send_bytes(self, content_type: str, content: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)

            def _stream_frames(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_sequence = -1
                try:
                    while True:
                        with stream.condition:
                            stream.condition.wait_for(
                                lambda: stream.sequence != last_sequence, timeout=2.0
                            )
                            frame = stream.frame
                            last_sequence = stream.sequence
                        if frame is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(frame)).encode("ascii")
                            + b"\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def update(self, frame, fps: float, objects: int) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return
        with self.condition:
            self.frame = encoded.tobytes()
            self.sequence += 1
            self.status = {"ready": True, "fps": round(fps, 1), "objects": objects}
            self.condition.notify_all()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def csi_pipeline(width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
    )


class YoloDetectorNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("yolo_detector")
        self.args = args
        self.bridge = CvBridge()
        self.det_pub = self.create_publisher(Detection2DArray, "/detections", 10)
        self.img_pub = self.create_publisher(Image, "/detection_image", 10)

        self.get_logger().info(f"加载模型: {args.weights}")
        self.model = YOLO(args.weights)
        self.names = self.model.names

        if args.source == "csi":
            self.cap = cv2.VideoCapture(
                csi_pipeline(args.width, args.height, args.fps), cv2.CAP_GSTREAMER
            )
        else:
            src = int(args.source) if args.source.isdigit() else args.source
            self.cap = cv2.VideoCapture(src)
            # UVC 摄像头默认 YUYV 未压缩，720p 下只能给 5~10 FPS，必须先切 MJPG 再设分辨率
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit(f"无法打开摄像头: {args.source}")
        self.get_logger().info(
            f"摄像头就绪: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )

        self.writer = None
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                args.save_video, fourcc, 15.0, (args.width, args.height)
            )
            self.get_logger().info(f"录制到: {args.save_video}")

        self.frame_times: deque = deque(maxlen=30)
        self.last_frame_time = None
        self.web = MjpegServer(args.web_port, args.jpeg_quality) if args.web_port else None
        if self.web is not None:
            self.get_logger().info(f"浏览器实时画面: http://0.0.0.0:{args.web_port}")
        self.timer = self.create_timer(1.0 / args.fps, self.tick)

    def tick(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("读取帧失败")
            return

        result = self.model.predict(
            frame, conf=self.args.conf, imgsz=self.args.imgsz, verbose=False
        )[0]
        now = time.perf_counter()
        if self.last_frame_time is not None:
            self.frame_times.append(now - self.last_frame_time)
        self.last_frame_time = now
        fps = len(self.frame_times) / max(sum(self.frame_times), 1e-6)

        stamp = self.get_clock().now().to_msg()
        msg = Detection2DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.frame_id

        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = self.names[cls_id]

            det = Detection2D()
            det.header = msg.header
            bbox = BoundingBox2D()
            bbox.center.position.x = (x1 + x2) / 2
            bbox.center.position.y = (y1 + y2) / 2
            bbox.size_x = x2 - x1
            bbox.size_y = y2 - y1
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = score
            det.results.append(hyp)
            msg.detections.append(det)

            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            text = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 4, int(y1)), color, -1
            )
            cv2.putText(
                frame, text, (int(x1) + 2, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

        cv2.putText(
            frame, f"FPS: {fps:.1f}  objs: {len(msg.detections)}", (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        self.det_pub.publish(msg)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))
        if self.writer is not None:
            self.writer.write(frame)
        if self.web is not None:
            self.web.update(frame, fps, len(msg.detections))

        if not self.args.no_display:
            cv2.imshow("YOLO detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def destroy_node(self) -> bool:
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if self.web is not None:
            self.web.close()
        if not self.args.no_display:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 实时检测 + ROS2 发布")
    parser.add_argument("--weights", default="best.engine")
    parser.add_argument("--source", default="0", help="摄像头编号、视频路径，或 csi")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15, help="目标帧率（定时器频率）")
    parser.add_argument("--frame_id", default="camera_link")
    parser.add_argument("--no-display", action="store_true", help="无显示器时使用")
    parser.add_argument("--web-port", type=int, default=0, help=">0 时开启浏览器实时画面")
    parser.add_argument("--jpeg-quality", type=int, default=80, choices=range(30, 96))
    parser.add_argument("--save-video", default="", help="录制结果视频的输出路径")
    args = parser.parse_args()

    rclpy.init()
    node = YoloDetectorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
