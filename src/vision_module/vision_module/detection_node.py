import rclpy
import cv2
import numpy as np
import pyrealsense2 as rs
import os, glob
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped,Pose
from ament_index_python.packages import get_package_prefix
from tf2_ros import TransformBroadcaster
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import LookupException
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import transformations as tf_transformations
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge

from vision_module.screw_detector import ScrewDetector
from vision_module.l_shape_detector import LShapeDetector
from vision_module.icp_fitter import ICPFITTER


def remove_duplicate_detections(results, threshold=15):
    filtered = []
    for r in results:
        if 'u' not in r or 'v' not in r:
            continue
        is_duplicate = any(
            abs(r['u'] - f['u']) < threshold and abs(r['v'] - f['v']) < threshold
            for f in filtered
        )
        if not is_duplicate:
            filtered.append(r)
    return filtered


class RealSenseVision(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.get_logger().info('✅ RealSense Detection Node Started')

        # subs / pubs
        self.subscription = self.create_subscription(String, '/detection_task', self.task_callback, 10)
        self.pose_pub = self.create_publisher(Pose, '/object_camera_pose', 10)
        self.alert_pub = self.create_publisher(String, '/orientation_anomaly', 10)
        self.image_pub = self.create_publisher(Image, '/color_image', 10)

        # ICP 指令 & 快取
        self.icp_region = "screw"
        self.icp_cmd_sub = self.create_subscription(String, '/icp_cmd', self.icp_cmd_callback, 10)
        self.latest_color = None
        self.latest_depth = None
        self.latest_intrin = None

        # flags
        self.screw_active = False
        self.lshape_active = False
        self.icp_fit_active = False

        # 只要「有下過任務指令」才算啟動過；用來避免一開程式就刷警告
        self.task_armed = False

        # realsense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # detectors
        source_root = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(source_root, 'template2_shape')
        self.screw_detector = ScrewDetector(template_path)
        self.l_templates = self.load_templates('template_l_shape')
        self.l_shape_detector = LShapeDetector(self.l_templates)
        self.icp_fitter = ICPFITTER()

        # TF + state
        self.br = TransformBroadcaster(self)
        self.prev_screw_results = []
        self.prev_avg_center = None
        self.prev_avg_pose = None
        self.frame_count = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 姿態異常檢測
        self.abnormal_counter = 0
        self.prev_yaw = 0.0
        self.prev_pitch = 0.0
        self.prev_roll = 0.0

        # 警告節流（避免狂刷）
        self.last_tf_warn_time = self.get_clock().now()

    def _warn_tf_rate_limited(self, text: str, min_interval_sec: float = 1.0):
        now = self.get_clock().now()
        if (now - self.last_tf_warn_time).nanoseconds * 1e-9 >= min_interval_sec:
            print(text)
            self.last_tf_warn_time = now

    def publish_image(self, color_image):
        resized_image = cv2.resize(color_image, (701, 481), interpolation=cv2.INTER_AREA)
        ros_image = CvBridge().cv2_to_imgmsg(resized_image, encoding="bgr8")
        # 這裡沒有 image_pub，就不發 ROS 影像；若需要可自行新增 publisher
        self.image_pub.publish(ros_image)
        pass

    def load_templates(self, folder_name, augment=False):
        source_root = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(source_root, folder_name)
        print(f"📂 嘗試讀取模板資料夾：{folder_path}")
        if not os.path.exists(folder_path):
            print(f"❌ 錯誤：資料夾不存在 - {folder_path}")
            return []
        paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        if len(paths) == 0:
            print(f"⚠️ 警告：找不到任何 .png 模板在資料夾 {folder_path}")
        templates = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append(img)
            else:
                print(f"⚠️ 無法讀取圖片：{path}")
        print(f"✅ 成功載入 {len(templates)} 張模板圖片")
        return templates

    def task_callback(self, msg: String):
        task = msg.data.lower().strip()
        self.screw_active = (task == 'screw')
        self.lshape_active = (task == 'l_shape')
        self.icp_fit_active = (task == 'icp_fit')
        self.task_armed = True  # ✅ 有收到任務指令才啟用後續 TF 檢查/警告
        self.get_logger().info(f"🔄 任務切換: {task}")

        if self.screw_active:
            self.abnormal_counter = 0
            # 切入螺絲模式時清空舊結果，避免誤以為已有偵測
            self.prev_screw_results = []

    def icp_cmd_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd in ("save screw", "screw save", "save_screw"):
            ok = self.icp_fitter.save_golden(self.latest_color, self.latest_depth, self.latest_intrin, "screw")
            self.get_logger().info("📸 儲存螺絲區 golden " + ("成功" if ok else "失敗"))
        elif cmd in ("save battery", "battery save", "save_battery"):
            ok = self.icp_fitter.save_golden(self.latest_color, self.latest_depth, self.latest_intrin, "battery")
            self.get_logger().info("📸 儲存電池區 golden " + ("成功" if ok else "失敗"))
        elif cmd in ("icp screw", "run screw", "icp_screw"):
            self.icp_region = "screw"
            self.get_logger().info("🔧 ICP 目標區域切到 screw")
        elif cmd in ("icp battery", "run battery", "icp_battery"):
            self.icp_region = "battery"
            self.get_logger().info("🔧 ICP 目標區域切到 battery")
        else:
            self.get_logger().warn(f"❓ 未知 icp_cmd: {cmd}")

    def broadcast_screw_tf(self, idx, x, y, z, quat=None):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = f'screw_{idx}'
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        if quat is None:
            quat = [0, 0, 0, 1]
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.br.sendTransform(t)

    def publish_avg_pose(self, x, y, z, quat):
        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]
        self.pose_pub.publish(msg)

    def check_angle_anomaly(self, yaw, pitch, roll, threshold=5.0):
        anomalies = []
        if abs(yaw - self.prev_yaw) > threshold:
            anomalies.append(f"Yaw Δ={yaw - self.prev_yaw:.2f}°")
        if abs(pitch - self.prev_pitch) > threshold:
            anomalies.append(f"Pitch Δ={pitch - self.prev_pitch:.2f}°")
        if abs(roll - self.prev_roll) > threshold:
            anomalies.append(f"Roll Δ={roll - self.prev_roll:.2f}°")
        if anomalies:
            msg = f"⚠️ Orientation anomaly: {', '.join(anomalies)} (thres=±{threshold}°)"
            self.get_logger().warn(msg)
            self.alert_pub.publish(String(data=msg))
        self.prev_yaw = yaw
        self.prev_pitch = pitch
        self.prev_roll = roll


def main():
    rclpy.init()
    node = RealSenseVision()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            frames = node.pipeline.wait_for_frames()
            aligned = node.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # 快取給 /icp_cmd 使用
            node.latest_color = color_image
            node.latest_depth = depth_image
            node.latest_intrin = depth_intrin

            # ---------- 螺絲模式 ----------
            if node.screw_active:
                have_fresh_detection = False
                screw_results = []

                # 每 10 幀做一次偵測
                if node.frame_count % 10 == 0:
                    results = node.screw_detector.detect(color_image, depth_frame)
                    results = remove_duplicate_detections(results)
                    results = sorted(results, key=lambda r: (r['u'], r['v']))
                    if len(results) == 4:
                        node.prev_screw_results = results
                        have_fresh_detection = True
                    screw_results = results
                else:
                    # 沒有新偵測 → 只用上一幀的結果畫圖，不廣播
                    screw_results = node.prev_screw_results

                # 繪製與（若有）姿態計算 + 廣播
                if len(screw_results) == 4:
                    avg_x = avg_y = avg_z = 0.0
                    avg_u = avg_v = 0

                    for i, r_ in enumerate(screw_results):
                        u, v = r_['u'], r_['v']
                        X, Y, Z = r_['X'], r_['Y'], r_['Z']
                        avg_x += X; avg_y += Y; avg_z += Z
                        avg_u += u;  avg_v += v
                        cv2.circle(color_image, (u, v), 6, (0, 255, 0), -1)
                        cv2.putText(color_image, f"X={X:.2f} Y={Y:.2f} Z={Z:.2f}",
                                    (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                        print(f"🟢 螺絲 {i+1}: (u={u}, v={v}), X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

                    # 只有「本幀有新偵測」才計算姿態 & 廣播 TF
                    if have_fresh_detection:
                        v0 = np.array([screw_results[0]['X'], screw_results[0]['Y'], screw_results[0]['Z']])
                        v1 = np.array([screw_results[1]['X'], screw_results[1]['Y'], screw_results[1]['Z']])
                        v2 = np.array([screw_results[2]['X'], screw_results[2]['Y'], screw_results[2]['Z']])
                        v3 = np.array([screw_results[3]['X'], screw_results[3]['Y'], screw_results[3]['Z']])
                        
                        

                        x_axis = v3 - v2
                        x_axis = x_axis / np.linalg.norm(x_axis)
                        temp_vec = v0 - v2
                        z_axis = np.cross(x_axis, temp_vec)
                        z_axis = z_axis / np.linalg.norm(z_axis)
                        y_axis = np.cross(z_axis, x_axis)

                        R_mat = np.column_stack((x_axis, y_axis, z_axis))
                        Tm = np.eye(4); Tm[:3,:3] = R_mat
                        quat = tf_transformations.quaternion_from_matrix(Tm)

                        r_e = R.from_quat(quat)
                        yaw_deg, pitch_deg, roll_deg = r_e.as_euler('zyx', degrees=True)

                        if node.prev_avg_pose is None:
                            node.prev_yaw = yaw_deg
                            node.prev_pitch = pitch_deg
                            node.prev_roll = roll_deg
                            print("First detect")
                        else:
                            yaw_diff   = abs(yaw_deg - node.prev_yaw)
                            pitch_diff = abs(pitch_deg - node.prev_pitch)
                            roll_diff  = abs(roll_deg - node.prev_roll)

                            if yaw_diff > 5 or pitch_diff > 5 or roll_diff > 5:
                                print(f"⚠️ 姿態變化過大！Yaw Δ={yaw_diff:.2f}°, Pitch Δ={pitch_diff:.2f}°, Roll Δ={roll_diff:.2f}° → 忽略此次結果")
                                node.abnormal_counter += 1
                                if node.abnormal_counter >= 10:
                                    print("❌ 已連續 10 次異常，顯示 NULL 並停止螺絲偵測")
                                    node.prev_avg_center = None
                                    node.prev_avg_pose = None
                                    node.screw_active = False
                                    node.abnormal_counter = 0
                                # 本幀不廣播
                                pass
                            else:
                                node.abnormal_counter = 0

                        # 廣播本幀 TF
                        for i, r_ in enumerate(screw_results):
                            node.broadcast_screw_tf(i + 1, r_['X'], r_['Y'], r_['Z'], quat)

                        # 平均中心 & 發布 Pose
                        avg_x /= 4.0; avg_y /= 4.0; avg_z /= 4.0
                        center_u = avg_u // 4; center_v = avg_v // 4

                        print(f"[INFO] Orientation: Yaw={yaw_deg:.1f}°, Pitch={pitch_deg:.1f}°, Roll={roll_deg:.1f}°")
                        node.prev_avg_center = (center_u, center_v)
                        node.prev_avg_pose = (avg_x, avg_y, avg_z, yaw_deg, pitch_deg, roll_deg)

                        node.publish_avg_pose(avg_x, avg_y, avg_z, quat)
                        node.check_angle_anomaly(yaw_deg, pitch_deg, roll_deg, threshold=5.0)

                # 疊上最後有效姿態資訊（僅視覺提示，不影響 TF）
                if node.prev_avg_center and node.prev_avg_pose:
                    center_u, center_v = node.prev_avg_center
                    avg_x, avg_y, avg_z, yaw_deg, pitch_deg, roll_deg = node.prev_avg_pose
                    cv2.circle(color_image, (int(center_u), int(center_v)), 8, (0,255,255), -1)
                    cv2.putText(color_image, f"Avg X={avg_x:.2f} Y={avg_y:.2f} Z={avg_z:.2f}",
                                (center_u - 100, center_v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv2.putText(color_image, f"Yaw={yaw_deg:.1f} Pitch={pitch_deg:.1f} Roll={roll_deg:.1f}",
                                (center_u - 100, center_v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # ---------- L-Shape 模式 ----------
            if node.lshape_active:
                results = node.l_shape_detector.detect_l_shape_lines(color_image)
                for cx, cy in results:
                    cv2.circle(color_image, (cx, cy), 5, (0,255,255), -1)

            # ---------- ICP 模式 ----------
            if node.icp_fit_active:
                dist = node.icp_fitter.icp_fit(color_image, depth_image, depth_intrin, region=node.icp_region)
                print(f"ICP Distance ({node.icp_region}): {dist:.4f}" if dist is not None else "ICP fitting failed")

            # ---------- 只在「任務已啟動」且「螺絲模式」時才做 TF 查詢或警告 ----------
            if node.task_armed and node.screw_active:
                if len(node.prev_screw_results) == 4:
                    # 有有效結果 → 查詢並輸出 TF 詳情
                    try:
                        now = Time()
                        for idx in range(1, 5):
                            tf = node.tf_buffer.lookup_transform(
                                target_frame='camera_link',
                                source_frame=f'screw_{idx}',
                                time=now,
                                timeout=Duration(seconds=0.5)
                            )
                            pos = tf.transform.translation
                            rot = tf.transform.rotation
                            print(f"[🔧 TF] screw_{idx} in camera_link:\n"
                                  f"  ↳ Position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})\n"
                                  f"  ↳ Quaternion: ({rot.x:.3f}, {rot.y:.3f}, {rot.z:.3f}, {rot.w:.3f})")
                    except LookupException:
                        # 理論上有廣播就不該進來；若進來，節流提示
                        node._warn_tf_rate_limited("⚠️ 無法查詢 TF: screw_X → camera_link")
                else:
                    # 沒抓到 4 顆 → 只在螺絲模式且任務已啟動時，節流提示
                    node._warn_tf_rate_limited("⚠️ 無法查詢 TF: screw_X → camera_link")

            # 顯示視窗
            cv2.imshow("RealSense Detection", color_image)
            node.publish_image(color_image)
            if cv2.waitKey(1) == ord('q'):
                break

            node.frame_count += 1

    finally:
        node.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
