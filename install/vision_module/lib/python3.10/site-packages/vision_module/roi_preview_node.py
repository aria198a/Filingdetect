import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseROIPreview(Node):

    def __init__(self):
        super().__init__('realsense_roi_preview')

        # ---- 可用 ROI 參數----
        self.declare_parameter('roi_top_left_x', 332)
        self.declare_parameter('roi_top_left_y', 262)
        self.declare_parameter('roi_bottom_right_x', 1113)
        self.declare_parameter('roi_bottom_right_y', 627)
        self.declare_parameter('line_thickness', 3)
        self.declare_parameter('font_scale', 0.8)
        self.declare_parameter('show_fps', True)
        self.declare_parameter('color_width', 1280)
        self.declare_parameter('color_height', 720)
        self.declare_parameter('fps', 30)

        self.roi_tl = (
            int(self.get_parameter('roi_top_left_x').value),
            int(self.get_parameter('roi_top_left_y').value)
        )
        self.roi_br = (
            int(self.get_parameter('roi_bottom_right_x').value),
            int(self.get_parameter('roi_bottom_right_y').value)
        )
        self.line_thickness = int(self.get_parameter('line_thickness').value)
        self.font_scale = float(self.get_parameter('font_scale').value)
        self.show_fps = bool(self.get_parameter('show_fps').value)
        self.color_w = int(self.get_parameter('color_width').value)
        self.color_h = int(self.get_parameter('color_height').value)
        self.fps = int(self.get_parameter('fps').value)
        self.period = max(1.0 / float(self.fps), 0.01)  # 固定頻率，避免 0.0 造成 busy loop

        # ---- Publisher ----
        self.info_pub = self.create_publisher(String, '/current_roi_info', 10)

        # ---- RealSense 初始化 ----
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.color_w, self.color_h, rs.format.z16, self.fps)

        self.align = rs.align(rs.stream.color)

        try:
            self.pipeline_profile = self.pipeline.start(self.config)
            self.get_logger().info('✅ RealSense 已啟動（彩色+深度，對齊至彩色）')
        except Exception as e:
            self.get_logger().error(f'❌ RealSense 啟動失敗：{e}')
            raise

        # 顯示一次 ROI 設定（不含寬高）
        self.log_and_publish_roi()

        # FPS
        self.last_tick = cv2.getTickCount()
        self.cur_fps = 0.0

        # 以固定 period 執行，不要 0.0
        self.timer = self.create_timer(self.period, self.loop_once)

        # 供 main() 偵測是否要結束
        self._should_quit = False

    def log_and_publish_roi(self):
        x1, y1 = self.roi_tl
        x2, y2 = self.roi_br
        msg = f'ROI tl=({x1},{y1}) br=({x2},{y2})'
        self.get_logger().info(f'📏 {msg}')
        self.info_pub.publish(String(data=msg))

    def draw_roi_overlay(self, img):
        x1, y1 = self.roi_tl
        x2, y2 = self.roi_br

        # 邊界裁切避免超出畫面
        h_img, w_img = img.shape[:2]
        x1 = int(np.clip(x1, 0, w_img - 1))
        y1 = int(np.clip(y1, 0, h_img - 1))
        x2 = int(np.clip(x2, 0, w_img - 1))
        y2 = int(np.clip(y2, 0, h_img - 1))

        # 確保左上 < 右下
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

        # 畫框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), self.line_thickness)

        # 只顯示座標
        info = f'ROI: ({x_min},{y_min})-({x_max},{y_max})'
        cv2.putText(img, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        # FPS
        if self.show_fps:
            cv2.putText(img, f'FPS: {self.cur_fps:.1f}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        return img

    def update_fps(self):
        now = cv2.getTickCount()
        dt = (now - self.last_tick) / cv2.getTickFrequency()
        self.last_tick = now
        if dt > 0:
            self.cur_fps = 1.0 / dt

    def loop_once(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                return

            color_image = np.asanyarray(color_frame.get_data())
            self.update_fps()

            vis = self.draw_roi_overlay(color_image.copy())

            cv2.imshow('ROI Preview (press s=save, q=quit)', vis)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                cv2.imwrite('roi_preview_snapshot.png', vis)
                self.get_logger().info('📸 已儲存 roi_preview_snapshot.png')
            elif key == ord('q'):
                self.get_logger().info('👋 收到退出指令(q)，準備關閉節點')
                self._should_quit = True  # 讓 main() 偵測後收尾
        except Exception as e:
            self.get_logger().warning(f'預覽更新失敗：{e}')

    def cleanup(self):
        """主程式要結束時呼叫，確保釋放資源。"""
        try:
            self.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    def destroy_node(self):
        # 也保險清一次
        self.cleanup()
        super().destroy_node()


def main():
    rclpy.init()
    node = RealSenseROIPreview()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node._should_quit:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
