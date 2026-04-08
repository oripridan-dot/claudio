"""
head_tracker.py — Real-Time 6DoF Head Tracker

Extracts head orientation (roll, pitch, yaw) and approximate translation
from MediaPipe Face Landmarker (478 landmarks) using a solvePnP-based approach.
Backward-compatible with legacy 468-landmark Face Mesh output.

Output: continuous quaternion stream fed to the HRTF engine via a lock-free
ring buffer (matching the SpatialLatencyGate requirement).

The Face Landmarker also exposes 52 blendshape coefficients for facial
expression tracking — useful for future avatar-driven mixing interfaces.

The quaternion stream runs independently of the gesture classifier —
head tracking is always active when a camera is available, even when
no named gestures are being executed.  This is what makes the holographic
binaural experience feel alive: you turn your head 2 degrees and the acoustic
field instantly follows.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class HeadPose:
    """6DoF head orientation + position estimate."""
    timestamp: float
    # Euler angles in degrees (for human-readable logging)
    yaw:   float   # left/right rotation
    pitch: float   # up/down rotation
    roll:  float   # tilt
    # Unit quaternion (w, x, y, z)
    quat:  tuple[float, float, float, float]
    # Normalised translation (0–1 relative to frame)
    tx: float = 0.0
    ty: float = 0.0


# 3D model points of key facial landmarks (in mm, generic head model)
_MODEL_POINTS = np.array([
    [0.0,       0.0,       0.0],      # Nose tip        (1)
    [0.0,      -330.0,    -65.0],     # Chin            (152)
    [-225.0,    170.0,    -135.0],    # Left eye corner (263)
    [225.0,     170.0,    -135.0],    # Right eye corner(33)
    [-150.0,   -150.0,    -125.0],    # Left mouth corner(287)
    [150.0,    -150.0,    -125.0],    # Right mouth corner(57)
], dtype=np.float64)

# Matching landmark indices in MediaPipe Face Mesh (468 points)
_FACE_INDICES = [1, 152, 263, 33, 287, 57]


def _euler_from_rotation_vector(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convert OpenCV rotation vector to (yaw, pitch, roll) in degrees."""
    rot_mat, _ = _rodrigues(rvec)
    sy = math.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-rot_mat[2, 0], sy))
        yaw   = math.degrees(math.atan2( rot_mat[1, 0], rot_mat[0, 0]))
        roll  = math.degrees(math.atan2( rot_mat[2, 1], rot_mat[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-rot_mat[2, 0], sy))
        yaw   = 0.0
        roll  = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1]))
    return yaw, pitch, roll


def _rodrigues(rvec: np.ndarray) -> tuple[np.ndarray, None]:
    """Pure-numpy Rodrigues rotation vector → 3×3 matrix."""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-8:
        return np.eye(3), None
    axis = rvec.flatten() / theta
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
    return R, None


def _rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3×3 rotation matrix to unit quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (w, x, y, z)


class SpatialHeadTracker:
    """
    Extracts 6DoF head pose from MediaPipe Face Mesh landmarks.
    Emits HeadPose objects at camera frame rate.

    The latest HeadPose is also written to a lock-free ring buffer
    (`quaternion_ring`) that the claudio-core HRTF thread reads from
    without mutex — satisfying the SpatialLatencyGate <1.5 ms update
    constraint.
    """

    RING_SIZE = 64

    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        self._fw = frame_width
        self._fh = frame_height
        # Approximate camera intrinsics (no calibration required for this use case)
        focal   = frame_width
        cx, cy  = frame_width / 2.0, frame_height / 2.0
        self._camera_matrix = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1 ],
        ], dtype=np.float64)
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        # Lock-free quaternion ring (written by camera thread, read by audio thread)
        self.quaternion_ring: deque[tuple[float, float, float, float]] = deque(
            maxlen=self.RING_SIZE
        )
        self._latest: HeadPose | None = None

    def update(self, face_landmarks: np.ndarray) -> HeadPose | None:
        """
        Ingest one Face Landmarker frame (478×3 or legacy 468×3 landmarks).
        Returns a HeadPose or None if estimation fails.
        """
        if face_landmarks is None:
            return None
        if face_landmarks.shape not in ((478, 3), (468, 3)):
            return None

        # Extract the 6 key landmarks and denormalise to pixel coords
        image_points = np.array([
            [face_landmarks[idx, 0] * self._fw,
             face_landmarks[idx, 1] * self._fh]
            for idx in _FACE_INDICES
        ], dtype=np.float64)

        # solvePnP — pure-numpy iterative solver (no OpenCV dependency)
        rvec, tvec = self._solve_pnp(image_points)
        if rvec is None:
            return None

        yaw, pitch, roll = _euler_from_rotation_vector(rvec)
        R, _ = _rodrigues(rvec)
        quat  = _rotation_matrix_to_quaternion(R)

        pose = HeadPose(
            timestamp=time.perf_counter(),
            yaw=yaw, pitch=pitch, roll=roll,
            quat=quat,
            tx=float(tvec[0] / self._fw),
            ty=float(tvec[1] / self._fh),
        )
        self._latest = pose
        # Push to lock-free ring — audio thread reads quat directly
        self.quaternion_ring.append(quat)
        return pose

    @property
    def latest_quaternion(self) -> tuple[float, float, float, float]:
        """Most recent head orientation quaternion. Safe to call from any thread."""
        if self.quaternion_ring:
            return self.quaternion_ring[-1]
        return (1.0, 0.0, 0.0, 0.0)  # identity

    def _solve_pnp(
        self, image_points: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Minimal iterative PnP solver using DLT initialisation.
        Returns (rvec, tvec) or (None, None) on failure.
        """
        try:
            # Direct Linear Transform initialisation
            n = len(_MODEL_POINTS)
            A = []
            for i in range(n):
                X, Y, Z    = _MODEL_POINTS[i]
                u, v        = image_points[i]
                fx = self._camera_matrix[0, 0]
                fy = self._camera_matrix[1, 1]
                cx = self._camera_matrix[0, 2]
                cy = self._camera_matrix[1, 2]
                A.append([X, Y, Z, 1, 0, 0, 0, 0,
                           -((u - cx)/fx)*X, -((u - cx)/fx)*Y,
                           -((u - cx)/fx)*Z, -((u - cx)/fx)])
                A.append([0, 0, 0, 0, X, Y, Z, 1,
                           -((v - cy)/fy)*X, -((v - cy)/fy)*Y,
                           -((v - cy)/fy)*Z, -((v - cy)/fy)])
            A_mat     = np.array(A, dtype=np.float64)
            _, _, Vt  = np.linalg.svd(A_mat)
            P         = Vt[-1].reshape(3, 4)
            R_est     = P[:, :3]
            # Orthogonalise via SVD
            U, _, Vt2 = np.linalg.svd(R_est)
            R_orth    = U @ Vt2
            if np.linalg.det(R_orth) < 0:
                U[:, 2] *= -1
                R_orth  = U @ Vt2
            tvec = P[:, 3] / (np.linalg.norm(R_est[:, 0]) + 1e-8)
            # Rotation matrix to axis-angle
            theta     = math.acos(
                max(-1.0, min(1.0, (np.trace(R_orth) - 1) / 2))
            )
            if abs(theta) < 1e-8:
                rvec = np.zeros((3, 1))
            else:
                axis = np.array([
                    R_orth[2, 1] - R_orth[1, 2],
                    R_orth[0, 2] - R_orth[2, 0],
                    R_orth[1, 0] - R_orth[0, 1],
                ]) / (2 * math.sin(theta))
                rvec = (theta * axis).reshape(3, 1)
            return rvec, tvec.reshape(3, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None, None
