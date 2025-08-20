import cv2
import numpy as np

class LShapeDetector:
    def __init__(self, templates,roi_boxes=None, scales=None, match_threshold=0.55):
        self.templates = templates
        self.l_match_threshold = 0.55
        self.l_roi_boxes = [
            (168, 103, 272, 563),
            (1192, 162, 1260, 527)
        ]

    def angle_between(self, v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        angle = np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))
        return angle

    def detect_l_shape_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        results = []

        for (x1, y1, x2, y2) in self.l_roi_boxes:
            roi = edges[y1:y2, x1:x2]
            lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=30, minLineLength=15, maxLineGap=15)

            if lines is not None:
                for i in range(len(lines)):
                    for j in range(i+1, len(lines)):
                        x11, y11, x12, y12 = lines[i][0]
                        x21, y21, x22, y22 = lines[j][0]

                        vec1 = np.array([x12 - x11, y12 - y11])
                        vec2 = np.array([x22 - x21, y22 - y21])
                        angle = self.angle_between(vec1, vec2)

                        if 80 <= angle <= 100:
                            x11g, y11g = x11 + x1, y11 + y1
                            x12g, y12g = x12 + x1, y12 + y1
                            x21g, y21g = x21 + x1, y21 + y1
                            x22g, y22g = x22 + x1, y22 + y1

                            cv2.line(frame, (x11g, y11g), (x12g, y12g), (255, 255, 0), 2)
                            cv2.line(frame, (x21g, y21g), (x22g, y22g), (255, 255, 0), 2)

                            cx = (x11g + x12g + x21g + x22g) // 4
                            cy = (y11g + y12g + y21g + y22g) // 4
                            results.append((cx, cy))
                            
        return results
