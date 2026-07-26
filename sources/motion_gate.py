"""Adaptive motion gate for camera streams — robust to wind/foliage.

Raw frame-differencing triggers on every swaying leaf. This uses an adaptive
Gaussian-mixture background model (MOG2) instead: repetitive background motion
and slow lighting changes are learned INTO the background, so only substantial,
sustained foreground — a person, a vehicle — reads as motion. Extra robustness:

- shadows (MOG2 marks them 127) are dropped, so moving cloud/sun shadows don't fire;
- blur + morphological open remove speckle noise from leaves/sensor grain;
- only the largest connected blob counts, against a minimum area fraction, so a
  person/car (one big blob) fires but scattered leaf pixels do not;
- motion must persist across several frames of the window, so a single gust doesn't.

Feed EVERY window's frames through `evaluate()` even when you end up skipping the
VLM — that keeps the background model learning the scene.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("home_assistant")


class MotionGate:
    def __init__(self, min_area_frac=0.008, min_motion_frames=3,
                 var_threshold=30, history=500, blur_ksize=5, warmup_frames=30):
        self.min_area_frac = float(min_area_frac)
        self.min_motion_frames = int(min_motion_frames)
        self.blur_ksize = int(blur_ksize) | 1  # force odd for GaussianBlur
        self.warmup_frames = int(warmup_frames)
        self._seen = 0
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=int(history), varThreshold=float(var_threshold),
            detectShadows=True)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _foreground_fraction(self, frame_rgb, learning_rate=-1):
        """Update the model with one frame; return its largest-blob area fraction."""
        gray = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2GRAY)
        if self.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        mask = self._bg.apply(gray, learningRate=learning_rate)
        # Keep only hard foreground (255); MOG2 shadows come back as 127.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        h, w = mask.shape[:2]
        largest = max(cv2.contourArea(c) for c in contours)
        return largest / float(h * w)

    def evaluate(self, frames, learning_rate=-1):
        """Feed a window of frames; return (moved, stats).

        During warmup the model is still untrained (everything looks like
        foreground), so we learn but never report motion.
        """
        motion_frames = 0
        max_frac = 0.0
        for f in frames:
            frac = self._foreground_fraction(f, learning_rate)
            max_frac = max(max_frac, frac)
            if frac >= self.min_area_frac:
                motion_frames += 1
            self._seen += 1
        warming = self._seen < self.warmup_frames
        moved = (not warming) and motion_frames >= self.min_motion_frames
        return moved, {
            "motion_frames": motion_frames,
            "frames": len(frames),
            "max_area_frac": round(max_frac, 4),
            "warming": warming,
        }
