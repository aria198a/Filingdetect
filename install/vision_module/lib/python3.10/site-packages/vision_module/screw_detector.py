import cv2
import numpy as np
import os
import glob
import pyrealsense2 as rs 

class ScrewDetector:
    def __init__(self, template_dir):
        self.templates = self.load_templates(template_dir)
        print(f"✅ Loaded {len(self.templates)} augmented templates from {template_dir}")

        # ROI 區域，可外部修改
        self.roi_top_left = (332, 262)
        self.roi_bottom_right = (1113, 627)

    def augment_template_rotation(self, img, angles=[-10, -5, 0, 5, 10]):
        augmented = []
        h, w = img.shape
        center = (w // 2, h // 2)
        for angle in angles:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
            augmented.append(rotated)
        return augmented

    def load_templates(self, folder_path):
        if not isinstance(folder_path, str):
            raise TypeError(f"❌ template_dir should be a string path, but got {type(folder_path)}")
        templates = []
        paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                augmented = self.augment_template_rotation(img)
                templates.extend(augmented)
            else:
                print(f"⚠️ Failed to load image from: {path}")
        return templates

    def match_templates(self, frame_gray, threshold=0.45):
        matches = []
        for template in self.templates:
            th, tw = template.shape
            res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                matches.append({
                    "pt": pt,
                    "size": (tw, th),
                    "score": res[pt[1], pt[0]]
                })
        return matches

    def remove_duplicates(self, matches, min_distance=20):
        filtered = []
        for m in matches:
            is_duplicate = False
            for f in filtered:
                dist = np.linalg.norm(np.array(m["pt"]) - np.array(f["pt"]))
                if dist < min_distance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(m)
        return filtered

    def detect(self, color_image, depth_frame):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        matches = self.match_templates(edges)
        filtered_matches = self.remove_duplicates(matches)

        results = []
        for m in filtered_matches:
            x, y = m["pt"]
            w, h = m["size"]
            center_u = x + w // 2
            center_v = y + h // 2

            if not (self.roi_top_left[0] <= center_u <= self.roi_bottom_right[0] and
                    self.roi_top_left[1] <= center_v <= self.roi_bottom_right[1]):
                continue

            Z = depth_frame.get_distance(center_u, center_v)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_u, center_v], Z)

            results.append({
                "u": center_u,
                "v": center_v,
                "X": X,
                "Y": Y,
                "Z": Z,
                "score": m["score"],
                "bbox": (x, y, w, h)
            })

        return results