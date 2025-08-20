import cv2
import pyrealsense2 as rs
import numpy as np
# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

print("請等待攝影機啟動畫面...")

# 讀取一張畫面供你選 ROI
for _ in range(30):
    frames = pipeline.wait_for_frames()

frame = pipeline.wait_for_frames()
color_frame = frame.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())

cv2.imshow("RealSense Frame (for ROI Selection)", color_image)
cv2.waitKey(1)

# 選兩個 ROI
print("請依序框選兩個 ROI（ENTER 確認，ESC 結束）")
roi1 = cv2.selectROI("Select ROI 1", color_image, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI 1")

roi2 = cv2.selectROI("Select ROI 2", color_image, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI 2")

pipeline.stop()

# 計算左上 & 右下座標
def roi_to_coords(roi):
    x, y, w, h = roi
    return (x, y), (x + w, y + h)

top_left1, bottom_right1 = roi_to_coords(roi1)
top_left2, bottom_right2 = roi_to_coords(roi2)

print("\n✅ 以下是你選取的兩個 ROI 框：")
print(f"ROI 1: 左上 {top_left1} 到 右下 {bottom_right1}")
print(f"ROI 2: 左上 {top_left2} 到 右下 {bottom_right2}")
