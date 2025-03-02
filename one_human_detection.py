import cv2
import numpy as np
from collections import deque


# Initialize video capture
cap = cv2.VideoCapture(0)

# Background subtraction with refined parameters
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Parameters for Farneback Optical Flow
fb_params = dict(pyr_scale=0.7, levels=5, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

# Initialize previous frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Store recent bounding boxes to stabilize detections
bbox_history = deque(maxlen=5)  # Stores last 5 detected bounding boxes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Background Subtraction
    fg_mask = bg_subtractor.apply(gray)
    
    # Morphological processing to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Compute Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
    
    # Convert Flow to Magnitude & Angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Motion thresholding (removes minor frame-to-frame noise)
    motion_mask = np.uint8(mag > 1.5) * 255

    # Combine Foreground and Motion Masks
    combined_mask = cv2.bitwise_and(fg_mask, motion_mask)

    # Find Contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.convexHull(c) for c in contours]  # Merge overlapping contours

    detected_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Ignore small objects
            continue

        # Compute bounding box & shape properties
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)  # Width-to-height ratio
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Adjust thresholds for full-body capture
        if 500 < area < 15000 and 0.5 < aspect_ratio < 5.0 and solidity > 0.4:
            # Expand bounding box slightly for full-body capture
            padding = 20
            x, y, w, h = (
                max(0, x - padding),
                max(0, y - padding),
                min(frame.shape[1] - x, w + 2 * padding),
                min(frame.shape[0] - y, h + 2 * padding)
            )

            detected_boxes.append((x, y, w, h))

    # **STABILIZATION LOGIC**: Only consider boxes that persist across frames
    if detected_boxes:
        bbox_history.append(detected_boxes)

    # Compute a **smooth bounding box** by averaging over last few frames
    if len(bbox_history) > 0:
        avg_x = int(np.mean([b[0] for boxes in bbox_history for b in boxes]))
        avg_y = int(np.mean([b[1] for boxes in bbox_history for b in boxes]))
        avg_w = int(np.mean([b[2] for boxes in bbox_history for b in boxes]))
        avg_h = int(np.mean([b[3] for boxes in bbox_history for b in boxes]))

        # Draw the smoothed bounding box
        cv2.rectangle(frame, (avg_x, avg_y), (avg_x + avg_w, avg_y + avg_h), (0, 255, 0), 2)

    # Update previous frame
    prev_gray = gray.copy()

    # Display the frame
    cv2.imshow('Refined Human Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()