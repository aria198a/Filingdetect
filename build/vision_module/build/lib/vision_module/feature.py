import cv2
import os
import pyrealsense2 as rs
import numpy as np

# ===== 固定儲存類型為 =====
mode = 'feature_shape'
template_dir = os.path.join(os.path.dirname(__file__), f'template_{mode}')
os.makedirs(template_dir, exist_ok=True)

# 自動編號
existing = [f for f in os.listdir(template_dir) if f.startswith(mode)]
template_count = len(existing)

# 初始化 RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

roi = None
drag = False
ix, iy = -1, -1

def mouse_callback(event, x, y, flags, param):
    global ix, iy, roi, drag
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        drag = True
    elif event == cv2.EVENT_MOUSEMOVE and drag:
        roi = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        roi = (ix, iy, x, y)
        drag = False

cv2.namedWindow("Canny View")
cv2.setMouseCallback("Canny View", mouse_callback)

print("🟦 請在 Canny 邊緣視圖中框選 L 型黑線樣本，按 S 儲存，ESC 離開。")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === Apply Canny edge detection for clearer view
        edges = cv2.Canny(gray, 50, 150)
        display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # 轉回彩色顯示紅框

        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow("Canny View", display)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == ord('s') and roi:
            x1, y1, x2, y2 = roi
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            cropped = edges[y1:y2, x1:x2]  
            
            filename = f"{mode}_{template_count}.png"
            save_path = os.path.join(template_dir, filename)
            cv2.imwrite(save_path, cropped)
            print(f"✅ 模板儲存成功：{save_path}")

            preview = cv2.resize(cropped, (min(300, cropped.shape[1]), min(300, cropped.shape[0])))
            cv2.imshow("Preview Saved", preview)
            cv2.waitKey(500)

            template_count += 1
            roi = None

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
