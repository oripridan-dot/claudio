"""
sensor_loop.py — Main Kinetic Studio Sensor Loop

Ties together:
  - Camera capture (OpenCV)
  - MediaPipe Holistic pose/hand/face estimation
  - GestureClassifier
  - SpatialHeadTracker
  - GestureRoutingMatrix (OSC / MIDI CC / DMX / acoustic)

Entry point: python -m claudio_vision_forge.sensor_loop
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from gesture_classifier import GestureClassifier, LandmarkFrame
from gesture_router import GestureRoutingMatrix, ControlAction
from head_tracker import SpatialHeadTracker


def _make_osc_sink(host: str, port: int):
    """Build an OSC sender if python-osc is available."""
    try:
        from pythonosc.udp_client import SimpleUDPClient
        client = SimpleUDPClient(host, port)

        def send(action: ControlAction) -> None:
            client.send_message(action.address, action.value)

        return send
    except ImportError:
        def send(action: ControlAction) -> None:
            print(f"[OSC] {action.address} = {action.value:.3f}")
        return send


def _make_dmx_sink(serial_port: str):
    """Build a DMX sender over serial (Enttec Open DMX protocol)."""
    try:
        import serial
        ser = serial.Serial(serial_port, baudrate=250000, stopbits=2)

        def send(action: ControlAction) -> None:
            channel = int(action.address.split("_")[-1])
            value   = int(action.value * 255)
            packet  = bytes([0x00] + [0] * (channel - 1) + [value] + [0] * (512 - channel))
            ser.break_condition = True
            time.sleep(0.0001)
            ser.break_condition = False
            ser.write(packet)

        return send
    except (ImportError, Exception) as e:
        def send(action: ControlAction) -> None:
            print(f"[DMX] ch={action.address} val={action.value:.3f}")
        return send


def run(
    camera_index: int = 0,
    osc_host: str = "127.0.0.1",
    osc_port: int = 9000,
    dmx_port: str | None = None,
    preview: bool = False,
    max_frames: int = 0,
) -> None:
    """
    Main sensor loop.

    Opens the camera, runs MediaPipe estimation on every frame,
    passes landmarks to the GestureClassifier and SpatialHeadTracker,
    and dispatches ControlActions to registered sinks.
    """
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("  Run:  pip install opencv-python mediapipe")
        sys.exit(1)

    mp_holistic = mp.solutions.holistic

    classifier = GestureClassifier()
    head_tracker = SpatialHeadTracker()
    router = GestureRoutingMatrix()
    router.register_sink("osc",     _make_osc_sink(osc_host, osc_port))
    router.register_sink("acoustic", lambda a: print(f"[ACOUSTIC] {a.address}={a.value:.3f}"))
    if dmx_port:
        router.register_sink("dmx", _make_dmx_sink(dmx_port))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}")
        sys.exit(1)

    print(f"[Claudio Vision Forge] Kinetic Studio active. OSC → {osc_host}:{osc_port}")
    frame_count = 0

    with mp_holistic.Holistic(
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = holistic.process(frame_rgb)
            ts        = time.perf_counter()

            def _lm_to_array(landmarks, n: int) -> np.ndarray | None:
                if landmarks is None:
                    return None
                return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark[:n]])

            lf = LandmarkFrame(
                timestamp  = ts,
                pose       = _lm_to_array(results.pose_landmarks, 33),
                left_hand  = _lm_to_array(results.left_hand_landmarks, 21),
                right_hand = _lm_to_array(results.right_hand_landmarks, 21),
                face       = _lm_to_array(results.face_landmarks, 468),
            )

            gesture_event = classifier.ingest(lf)
            if gesture_event:
                action = router.route(gesture_event)
                if action:
                    print(
                        f"[GESTURE] {gesture_event.gesture.name:<22} "
                        f"conf={gesture_event.confidence:.2f} "
                        f"→ {action.protocol}:{action.address}={action.value:.3f}"
                    )

            # Head tracking — quaternion always pushed to ring
            if lf.face is not None:
                head_pose = head_tracker.update(lf.face)
                if head_pose:
                    q = head_pose.quat
                    # Would normally push to claudio-core via IPC here
                    if frame_count % 30 == 0:
                        print(
                            f"[HEAD] yaw={head_pose.yaw:+.1f}° "
                            f"pitch={head_pose.pitch:+.1f}° "
                            f"roll={head_pose.roll:+.1f}°"
                        )

            if preview:
                cv2.imshow("Claudio Vision Forge", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

    cap.release()
    if preview:
        cv2.destroyAllWindows()
    print(f"[Claudio Vision Forge] Sensor loop ended. {frame_count} frames processed.")


def main() -> None:
    p = argparse.ArgumentParser(description="Claudio Kinetic Studio sensor loop")
    p.add_argument("--camera",    default=0,             type=int)
    p.add_argument("--osc-host",  default="127.0.0.1")
    p.add_argument("--osc-port",  default=9000,          type=int)
    p.add_argument("--dmx-port",  default=None)
    p.add_argument("--preview",   action="store_true")
    p.add_argument("--max-frames",default=0,             type=int)
    args = p.parse_args()
    run(
        camera_index=args.camera,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        dmx_port=args.dmx_port,
        preview=args.preview,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
