"""
Core Detection & Counting Engine for Head & Person Counting.
Uses Ultralytics YOLO with real-time bounding box extraction and HUD annotation.
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO


class HeadCounter:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initialize the HeadCounter with a YOLO model.
        Args:
            model_name (str): YOLO model path or name (default 'yolov8n.pt').
        """
        self.model_name = model_name
        self.model = YOLO(model_name)
        # Class 0 in standard COCO is 'person'
        self.person_class_id = 0
        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        detect_mode: str = "head",
        draw: bool = True,
    ):
        """
        Process a single image frame, detect people/heads, and optionally draw HUD.

        Args:
            frame (np.ndarray): Input image/frame in BGR format.
            conf_threshold (float): Minimum confidence threshold (0.0 to 1.0).
            detect_mode (str): 'head' (focuses on head/upper-body region) or 'person' (full body).
            draw (bool): Whether to draw bounding boxes and HUD on the frame.

        Returns:
            annotated_frame (np.ndarray): Frame with annotations.
            count (int): Number of detected heads/persons.
            detections (list): List of dicts with bbox coordinates and confidences.
        """
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff) if self.fps > 0 else (1.0 / time_diff)
        self.prev_time = current_time

        # Run inference (only detect 'person' class 0)
        results = self.model.predict(
            source=frame,
            classes=[self.person_class_id],
            conf=conf_threshold,
            verbose=False,
        )

        detections = []
        annotated_frame = frame.copy() if draw else frame
        h, w = frame.shape[:2]

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy

                if detect_mode == "head":
                    # Head region approximation: top 30% of detected person bounding box
                    box_h = y2 - y1
                    box_w = x2 - x1
                    head_y2 = y1 + box_h * 0.32
                    
                    # Make head box slightly tighter or proportional
                    head_x1 = max(0, int(x1 + box_w * 0.1))
                    head_x2 = min(w, int(x2 - box_w * 0.1))
                    head_y1 = max(0, int(y1))
                    head_y2 = min(h, int(head_y2))
                    
                    # If head area is valid
                    if head_x2 > head_x1 and head_y2 > head_y1:
                        detections.append({
                            "bbox": (head_x1, head_y1, head_x2, head_y2),
                            "confidence": conf,
                            "type": "head",
                            "full_person_bbox": (int(x1), int(y1), int(x2), int(y2)),
                        })
                else:
                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": conf,
                        "type": "person",
                        "full_person_bbox": (int(x1), int(y1), int(x2), int(y2)),
                    })

        count = len(detections)

        if draw:
            annotated_frame = self._draw_annotations(
                annotated_frame, detections, count, detect_mode
            )

        return annotated_frame, count, detections

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: list,
        count: int,
        detect_mode: str,
    ) -> np.ndarray:
        """
        Draw modern styling HUD, bounding boxes, labels, and count badge.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw individual detection bounding boxes
        for i, det in enumerate(detections):
            bx1, by1, bx2, by2 = det["bbox"]
            conf = det["confidence"]

            if detect_mode == "head":
                # Draw subtle person body silhouette box if in head mode
                px1, py1, px2, py2 = det["full_person_bbox"]
                cv2.rectangle(
                    overlay,
                    (px1, py1),
                    (px2, py2),
                    (100, 100, 100),
                    1,
                    lineType=cv2.LINE_AA,
                )

                # Draw glowing head box
                box_color = (0, 220, 255)  # Cyan/Gold in BGR
                cv2.rectangle(
                    overlay,
                    (bx1, by1),
                    (bx2, by2),
                    box_color,
                    2,
                    lineType=cv2.LINE_AA,
                )

                # Draw corner brackets for high-tech look
                line_len = min(15, (bx2 - bx1) // 3, (by2 - by1) // 3)
                # Top-Left
                cv2.line(overlay, (bx1, by1), (bx1 + line_len, by1), (0, 255, 128), 3)
                cv2.line(overlay, (bx1, by1), (bx1, by1 + line_len), (0, 255, 128), 3)
                # Top-Right
                cv2.line(overlay, (bx2, by1), (bx2 - line_len, by1), (0, 255, 128), 3)
                cv2.line(overlay, (bx2, by1), (bx2, by1 + line_len), (0, 255, 128), 3)
                # Bottom-Left
                cv2.line(overlay, (bx1, by2), (bx1 + line_len, by2), (0, 255, 128), 3)
                cv2.line(overlay, (bx1, by2), (bx1, by2 - line_len), (0, 255, 128), 3)
                # Bottom-Right
                cv2.line(overlay, (bx2, by2), (bx2 - line_len, by2), (0, 255, 128), 3)
                cv2.line(overlay, (bx2, by2), (bx2, by2 - line_len), (0, 255, 128), 3)

                label = f"Head #{i+1} ({conf:.2f})"
                label_y = max(18, by1 - 6)
                cv2.putText(
                    overlay,
                    label,
                    (bx1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )
            else:
                # Full Person Box
                box_color = (0, 200, 255)
                cv2.rectangle(
                    overlay,
                    (bx1, by1),
                    (bx2, by2),
                    box_color,
                    2,
                    lineType=cv2.LINE_AA,
                )
                label = f"Person #{i+1} ({conf:.2f})"
                label_y = max(18, by1 - 6)
                cv2.putText(
                    overlay,
                    label,
                    (bx1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

        # Draw HUD Header Banner at top
        banner_height = 55
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 25), -1)
        cv2.line(overlay, (0, banner_height), (w, banner_height), (0, 200, 255), 2)

        # Alpha blend HUD background slightly for sleek effect
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        # Re-draw text cleanly on final blended frame
        mode_str = "HEADS" if detect_mode == "head" else "PERSONS"
        count_text = f"TOTAL {mode_str}: {count}"
        fps_text = f"FPS: {self.fps:.1f}"

        # Count badge text
        cv2.putText(
            frame,
            count_text,
            (20, 37),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            (0, 255, 128),
            2,
            lineType=cv2.LINE_AA,
        )

        # FPS indicator
        cv2.putText(
            frame,
            fps_text,
            (w - 140, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2,
            lineType=cv2.LINE_AA,
        )

        return frame
