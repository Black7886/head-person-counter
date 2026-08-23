"""
Command-Line Runner for Real-Time Head & Person Counter.
Run with a webcam, an RTSP feed, or a pre-recorded video file.

Usage:
  python detect.py --source 0                    # Webcam
  python detect.py --source video.mp4            # Video file
  python detect.py --source 0 --conf 0.45        # Custom confidence
  python detect.py --source test.mp4 --save out.mp4  # Save output
"""

import argparse
import os
import sys
import time
import cv2
from counter import HeadCounter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Head & Person Counter using YOLO"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: '0' or webcam index, or path to video file (.mp4, .avi, etc.)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold (0.0 to 1.0, default: 0.35)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["head", "person"],
        default="head",
        help="Detection mode: 'head' (head region) or 'person' (full body)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model checkpoint (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save output video (e.g., output.mp4)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display OpenCV preview window (useful for headless processing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input source
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"[*] Initializing model: {args.model} ...")
    counter = HeadCounter(model_name=args.model)

    print(f"[*] Opening video source: {args.source} ...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video source '{args.source}'.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[*] Stream Properties: {width}x{height} @ {input_fps:.1f} FPS")
    print(f"[*] Mode: {args.mode.upper()} | Confidence: {args.conf}")
    print("[*] Press 'q' or 'ESC' on the video window to stop.")

    # Setup video writer if save is requested
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, input_fps, (width, height))
        print(f"[*] Output will be saved to: {args.save}")

    frame_idx = 0
    max_count = 0
    total_count_sum = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[*] End of video stream reached.")
                break

            frame_idx += 1

            # Process frame with counter
            annotated_frame, count, detections = counter.process_frame(
                frame=frame,
                conf_threshold=args.conf,
                detect_mode=args.mode,
                draw=True,
            )

            # Update stats
            max_count = max(max_count, count)
            total_count_sum += count

            # Write to output file if enabled
            if writer is not None:
                writer.write(annotated_frame)

            # Display window
            if not args.no_show:
                cv2.imshow("Head & Person Counter [Press Q to Exit]", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("\n[*] User interrupted execution.")
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    avg_count = total_count_sum / frame_idx if frame_idx > 0 else 0

    print("\n" + "=" * 50)
    print("             PROCESSING SUMMARY")
    print("=" * 50)
    print(f"  Total Frames Processed : {frame_idx}")
    print(f"  Elapsed Time           : {elapsed:.2f} s")
    print(f"  Average FPS            : {avg_fps:.1f}")
    print(f"  Peak Count Observed    : {max_count}")
    print(f"  Average Count / Frame  : {avg_count:.2f}")
    if args.save:
        print(f"  Saved Output File      : {os.path.abspath(args.save)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
