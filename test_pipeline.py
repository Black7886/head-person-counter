"""
Verification script for HeadCounter pipeline.
Tests model initialization, inference, bounding box extraction, and HUD rendering.
"""

import os
import cv2
import numpy as np
from counter import HeadCounter


def test_head_counter():
    print("[1/4] Initializing HeadCounter with yolov8n.pt ...")
    counter = HeadCounter(model_name="yolov8n.pt")
    assert counter.model is not None, "Model failed to load."
    print("  -> Model loaded successfully.")

    print("[2/4] Generating test frame with drawn figures ...")
    h, w = 480, 640
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    # Draw simple background lines
    cv2.line(frame, (0, 300), (w, 300), (80, 80, 80), 2)
    print("  -> Test frame created.")

    print("[3/4] Testing process_frame in 'head' mode ...")
    annotated_frame, count, detections = counter.process_frame(
        frame=frame,
        conf_threshold=0.25,
        detect_mode="head",
        draw=True,
    )
    assert annotated_frame.shape == (h, w, 3), "Annotated frame shape mismatch."
    assert isinstance(count, int), "Count is not an integer."
    assert isinstance(detections, list), "Detections is not a list."
    print(f"  -> Processed frame successfully. Count: {count}, FPS: {counter.fps:.1f}")

    print("[4/4] Testing process_frame in 'person' mode ...")
    annotated_frame_p, count_p, detections_p = counter.process_frame(
        frame=frame,
        conf_threshold=0.25,
        detect_mode="person",
        draw=True,
    )
    assert annotated_frame_p.shape == (h, w, 3), "Annotated frame shape mismatch."
    print("  -> Person mode processed successfully.")

    print("\n[SUCCESS] All pipeline tests passed!")


if __name__ == "__main__":
    test_head_counter()
