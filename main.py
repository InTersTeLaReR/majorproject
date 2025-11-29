import cv2
from ultralytics import YOLO, solutions

# ----------------------------
# 1. Open video file
# ----------------------------
video_file = r"C:\Users\Santosh TD\OneDrive - TECHNODYSIS PRIVATE LIMITED\Desktop\mp\zone_breach.mp4"
cap = cv2.VideoCapture(video_file)
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
video_writer = cv2.VideoWriter(
    "vision-eye-priority-zone.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ----------------------------
# 2. Initialize YOLO model and VisionEye helper
# ----------------------------
model = YOLO("yolo11n.pt")

visioneye = solutions.VisionEye(
    show=False,              # Disable window display
    model=model,
    classes=[0, 2, 3, 5, 7]  # person(0), car(2), motorbike(3), bus(5), truck(7)
)

# ----------------------------
# 3. Define static priority scores
# ----------------------------
STATIC_PRIORITY = {
    "person": 100,
    "bicycle": 80,
    "car": 60,
    "motorbike": 50,
    "bus": 40,
    "truck": 30,
    "traffic light": 20,
    "stop sign": 15,
    "cone": 10,
    "unknown": 5,
}

# ----------------------------
# 4. Define zone (upper 20% of frame)
# ----------------------------
zone_top_left = (int(w * 0.25), int(h * 0.05))
zone_bottom_right = (int(w * 0.75), int(h * 0.25))

# ----------------------------
# 5. Process each video frame
# ----------------------------
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video processing complete.")
        break

    # Run YOLO detection
    results = model(im0, verbose=False)

    # Optional: VisionEye overlay
    _ = visioneye(im0)

    # Draw the zone rectangle (upper section)
    cv2.rectangle(im0, zone_top_left, zone_bottom_right, (0, 0, 255), 2)
    cv2.putText(im0, "Zone", (zone_top_left[0] + 10, zone_top_left[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Iterate over detected objects
    for result in results:
        for det in result.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls = int(det.cls)
            label = model.names[cls]

            # ----------------------------
            # Static priority assignment
            # ----------------------------
            priority_score = STATIC_PRIORITY.get(label, 0)

            # Draw bounding box and static priority
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im0, f"{label}", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(im0, f"Priority: {priority_score}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # ----------------------------
            # Zone monitoring (upper section)
            # ----------------------------
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if zone_top_left[0] <= cx <= zone_bottom_right[0] and \
               zone_top_left[1] <= cy <= zone_bottom_right[1]:
                cv2.putText(im0, "Zone Breach!", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save frame to output video
    video_writer.write(im0)

# ----------------------------
# 6. Release resources
# ----------------------------
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("✅ Processing finished. Output saved as 'vision-eye-priority-zone.avi'")
