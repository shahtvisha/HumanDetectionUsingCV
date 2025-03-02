import cv2
import numpy as np
import json
from collections import deque

def merge_boxes(boxes, overlap_threshold=0.3):
    merged_boxes = []
    taken = set()
    
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if i in taken:
            continue
        
        new_x, new_y, new_w, new_h = x1, y1, w1, h1
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i != j and j not in taken:
                xa, ya = max(new_x, x2), max(new_y, y2)
                xb, yb = min(new_x + new_w, x2 + w2), min(new_y + new_h, y2 + h2)
                inter_area = max(0, xb - xa) * max(0, yb - ya)
                
                box1_area = new_w * new_h
                box2_area = w2 * h2
                union_area = box1_area + box2_area - inter_area
                
                if inter_area / union_area > overlap_threshold:
                    new_x = min(new_x, x2)
                    new_y = min(new_y, y2)
                    new_w = max(new_x + new_w, x2 + w2) - new_x
                    new_h = max(new_y + new_h, y2 + h2) - new_y
                    taken.add(j)
        
        merged_boxes.append((new_x, new_y, new_w, new_h))
    return merged_boxes

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
fb_params = dict(pyr_scale=0.7, levels=5, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

ret, prev_frame = cap.read()
while not ret:
    ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

tracked_objects = {}
bbox_history = deque(maxlen=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_subtractor.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = np.uint8(mag > 1.5) * 255
    combined_mask = cv2.bitwise_and(fg_mask, motion_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        if 500 < area < 15000 and 0.5 < w / float(h) < 5.0 and solidity > 0.4:
            padding = 20
            x, y, w, h = (
                max(0, x - padding),
                max(0, y - padding),
                min(frame.shape[1] - x, w + 2 * padding),
                min(frame.shape[0] - y, h + 2 * padding)
            )
            detected_boxes.append((x, y, w, h))
    
    detected_boxes = merge_boxes(detected_boxes)
    updated_tracked_objects = {}
    
    for box in detected_boxes:
        x, y, w, h = box
        matched = False
        
        for obj_id, (ox, oy, ow, oh) in tracked_objects.items():
            if abs(x - ox) < 40 and abs(y - oy) < 40:
                updated_tracked_objects[obj_id] = box
                matched = True
                break
        
        if not matched:
            new_id = len(updated_tracked_objects) + 1
            updated_tracked_objects[new_id] = box
    
    tracked_objects = updated_tracked_objects
    bbox_history.append(tracked_objects.copy())
    num_people = len(tracked_objects)

    for (obj_id, (x, y, w, h)) in tracked_objects.items():
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"ID {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # cv2.putText(frame, f"People Detected: {num_people}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    prev_gray = gray.copy()
    
    tracking_data = {"frame": len(bbox_history), "tracked_objects": tracked_objects}
    with open("tracking_data.json", "a") as json_file:
        json.dump(tracking_data, json_file)
        json_file.write("\n")
    
    cv2.imshow('Multi-Human Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
