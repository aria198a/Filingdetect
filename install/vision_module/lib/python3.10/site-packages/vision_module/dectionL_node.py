#!/usr/bin/env python3
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyrealsense2 as rs

# ---------- 角點挑選（依象限偏好） ----------
def harris_corner_pref(gray, pref='LB'):
    g = cv2.GaussianBlur(gray, (5,5), 0)
    h = cv2.cornerHarris(np.float32(g)/255.0, 2, 3, 0.04)
    h = cv2.dilate(h, None)
    ys, xs = np.where(h > 0.01 * h.max())
    if len(xs) == 0:
        return (gray.shape[1]//2, gray.shape[0]//2)
    if   pref == 'LB': bias = (-xs + ys)
    elif pref == 'RB': bias = ( xs + ys)
    elif pref == 'LT': bias = (-xs - ys)
    else:              bias = ( xs - ys)  # RT
    scores = bias + 0.001 * h[ys, xs]
    i = np.argmax(scores)
    return (int(xs[i]), int(ys[i]))

def refine_peak(res, loc):
    x, y = loc
    if x <= 0 or y <= 0 or x >= res.shape[1]-1 or y >= res.shape[0]-1:
        return (float(x), float(y))
    Z = res[y-1:y+2, x-1:x+2].astype(np.float32)
    denom_x = (Z[1,2] - 2*Z[1,1] + Z[1,0])
    denom_y = (Z[2,1] - 2*Z[1,1] + Z[0,1])
    dx = 0.5 * (Z[1,2] - Z[1,0]) / (denom_x + 1e-9)
    dy = 0.5 * (Z[2,1] - Z[0,1]) / (denom_y + 1e-9)
    return (x + float(dx), y + float(dy))

def quat_from_R(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = math.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2]) * 2.0
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = math.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2]) * 2.0
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return (qx, qy, qz, qw)

class LShapeDetectorNode(Node):
    """
    /lshape/cmd：
      - capture_left / capture_right / capture_screw
      - reset_left   / reset_right   / reset_screw
      - save / quit
    """
    def __init__(self):
        super().__init__('lshape_detector')

        # 影像 & 頻率
        self.declare_parameter('color_width', 1280)
        self.declare_parameter('color_height', 720)
        self.declare_parameter('fps', 30)

        # 顯示
        self.declare_parameter('line_thickness', 3)
        self.declare_parameter('font_scale', 0.8)
        self.declare_parameter('show_fps', True)

        # Canny
        self.declare_parameter('canny_low', 60)
        self.declare_parameter('canny_high', 160)
        self.declare_parameter('gauss_ksize', 5)

        # 模板比對
        self.declare_parameter('tm_alpha', 0.5)
        self.declare_parameter('tm_threshold', 0.55)
        self.declare_parameter('tm_update_threshold', 0.60)
        self.declare_parameter('search_margin', 100)

        # ROI 跳動防護
        self.declare_parameter('max_jump_px', 80)

        # 回退與節流
        self.declare_parameter('miss_limit', 10)
        self.declare_parameter('log_every_n', 10)
        self.declare_parameter('pub_every_n', 1)

        # 讀參數
        self.W = int(self.get_parameter('color_width').value)
        self.H = int(self.get_parameter('color_height').value)
        self.FPS = int(self.get_parameter('fps').value)
        self.lt = int(self.get_parameter('line_thickness').value)
        self.fs = float(self.get_parameter('font_scale').value)
        self.show_fps = bool(self.get_parameter('show_fps').value)
        self.canny_low = int(self.get_parameter('canny_low').value)
        self.canny_high = int(self.get_parameter('canny_high').value)
        self.gk = int(self.get_parameter('gauss_ksize').value);  self.gk = self.gk if self.gk%2==1 else 5
        self.alpha = float(self.get_parameter('tm_alpha').value)
        self.thr = float(self.get_parameter('tm_threshold').value)
        self.thr_upd = float(self.get_parameter('tm_update_threshold').value)
        self.margin = int(self.get_parameter('search_margin').value)
        self.max_jump = float(self.get_parameter('max_jump_px').value)
        self.miss_limit = int(self.get_parameter('miss_limit').value)
        self.log_every_n = int(self.get_parameter('log_every_n').value)
        self.pub_every_n = int(self.get_parameter('pub_every_n').value)

        # Pub/Sub
        self.info_pub = self.create_publisher(String, '/lshape_corner_info', 10)
        self.tf_pub = self.create_publisher(String, '/lshape_tf', 10)  # 也會塞兩個目標點 XYZ
        self.cmd_sub = self.create_subscription(String, '/lshape/cmd', self.on_cmd, 10)

        # RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, self.FPS)
        self.config.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, self.FPS)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        self.get_logger().info('✅ RealSense 已啟動（彩色+深度，對齊至彩色）')

        # 內參（啟動後第一幀取得）
        self.fx = self.fy = self.cx = self.cy = None

        # 視窗
        self.win = "L/R/Screw to 1-Point (X=R→L)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 960, 540)

        # 狀態
        self.last_tick = cv2.getTickCount()
        self.cur_fps = 0.0
        self.timer = self.create_timer(max(1.0/self.FPS, 0.01), self.loop_once)
        self._should_quit = False
        self._save_flag = False
        self.frame_idx = 0

        # 三側模板/狀態 + 儲存兩種目標點
        self.state = { 'left': self._new_side_state(), 'right': self._new_side_state(), 'screw': self._new_side_state() }
        self.goal1_cam = None; self.goal1_uv = None  # 目標點一：三點平均
        self.goal2_cam = None; self.goal2_uv = None  # 目標點二：投影到 X 軸
        self._last_color = None

    def _new_side_state(self):
        return dict(tmpl_gray=None, tmpl_edge=None, tmpl_corner=None,
                    prev_top_left=None, miss_count=0, last_xyz=None, last_ok=False)

    # ---------- 指令 ----------
    def on_cmd(self, msg: String):
        data = (msg.data or "").strip().lower()
        if data in ('capture_left','capture_right','capture_screw'):
            side = 'left' if 'left' in data else ('right' if 'right' in data else 'screw')
            try: color = self._last_color.copy()
            except Exception:
                self.get_logger().warning("還沒取得影像，稍後再試。"); return
            self.capture_roi_as_template(color, side)
        elif data in ('reset_left','reset_right','reset_screw'):
            side = 'left' if 'left' in data else ('right' if 'right' in data else 'screw')
            self.clear_template(side)
        elif data == 'save':
            self._save_flag = True
        elif data == 'quit':
            self.get_logger().info("收到 quit 指令。"); self._should_quit = True

    def clear_template(self, side):
        st = self.state[side]
        st.update(dict(tmpl_gray=None, tmpl_edge=None, tmpl_corner=None,
                       prev_top_left=None, miss_count=0, last_xyz=None, last_ok=False))
        self.get_logger().info(f"🧹 已清除 {side.upper()} 模板。")

    def capture_roi_as_template(self, color_bgr, side):
        title = f"框 {side.upper()} ROI (Enter 確認 / ESC 取消)"
        box = cv2.selectROI(title, color_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(title)
        x,y,w,h = [int(v) for v in box]
        if w<=0 or h<=0:
            self.get_logger().warning("未選擇 ROI，忽略。"); return
        patch = color_bgr[y:y+h, x:x+w]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.gk,self.gk), 0)
        edge = cv2.Canny(blur, self.canny_low, self.canny_high)

        if side == 'screw':
            u_refined = w/2.0; v_refined = h/2.0
        else:
            pref = 'LB' if side=='left' else 'RB'  # 左抓左下，右抓右下
            u0, v0 = harris_corner_pref(gray, pref=pref)
            corners = np.array([[float(u0), float(v0)]], dtype=np.float32).reshape(-1,1,2)
            try:
                cv2.cornerSubPix(gray, corners, (7,7), (-1,-1),
                                 (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-2))
                u_refined = float(corners[0,0,0]); v_refined = float(corners[0,0,1])
            except Exception:
                u_refined, v_refined = float(u0), float(v0)

        st = self.state[side]
        st['tmpl_gray'] = gray; st['tmpl_edge'] = edge
        st['tmpl_corner'] = (u_refined, v_refined)
        st['prev_top_left'] = (x, y); st['miss_count'] = 0
        st['last_xyz'] = None; st['last_ok'] = False

        cv2.imwrite(f'template_{side}_gray.png', gray)
        cv2.imwrite(f'template_{side}_edge.png', edge)
        self.get_logger().info(f"🎯 {side.upper()} 模板：rect=({x},{y},{w},{h}) corner=({u_refined:.2f},{v_refined:.2f})")

    # ---------- 工具 ----------
    def update_fps(self):
        now = cv2.getTickCount()
        dt = (now - self.last_tick) / cv2.getTickFrequency()
        self.last_tick = now
        if dt > 0: self.cur_fps = 1.0 / dt

    def depth_median(self, depth_frame, u, v, win=5):
        h = depth_frame.get_height();  w = depth_frame.get_width()
        half = max(1, win // 2)
        uu, vv = int(round(u)), int(round(v))
        xs = range(max(0, uu-half), min(w, uu+half+1))
        ys = range(max(0, vv-half), min(h, vv+half+1))
        vals = []
        for yy in ys:
            for xx in xs:
                d = depth_frame.get_distance(xx, yy)
                if d > 0: vals.append(d)
        return float(np.median(vals)) if vals else 0.0

    def choose_search_roi(self, side, full_shape):
        st = self.state[side]; H, W = full_shape[:2]
        if st['tmpl_gray'] is None or st['prev_top_left'] is None or st['miss_count'] >= self.miss_limit:
            return (0,0,W,H)
        x, y = st['prev_top_left']; th, tw = st['tmpl_gray'].shape[:2]
        xm = max(0, int(round(x)) - self.margin)
        ym = max(0, int(round(y)) - self.margin)
        xM = min(W, int(round(x)) + tw + self.margin)
        yM = min(H, int(round(y)) + th + self.margin)
        return (xm, ym, xM - xm, yM - ym)

    def match_template(self, side, gray_full, edge_full):
        st = self.state[side]; th, tw = st['tmpl_gray'].shape[:2]
        sx, sy, sw, sh = self.choose_search_roi(side, gray_full.shape)
        search_g = gray_full[sy:sy+sh, sx:sx+sw]; search_e = edge_full[sy:sy+sh, sx:sx+sw]
        if search_g.shape[0] < th or search_g.shape[1] < tw: return None, None
        res_g = cv2.matchTemplate(search_g, st['tmpl_gray'], cv2.TM_CCOEFF_NORMED)
        res_e = cv2.matchTemplate(search_e, st['tmpl_edge'], cv2.TM_CCOEFF_NORMED)
        res = self.alpha * res_g + (1.0 - self.alpha) * res_e
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        subx, suby = refine_peak(res, max_loc)
        return (sx + subx, sy + suby), float(max_val)

    def backproject(self, u, v, z):
        if None in (self.fx, self.fy, self.cx, self.cy) or z <= 0: return None
        X = (u - self.cx) / self.fx * z
        Y = (v - self.cy) / self.fy * z
        return np.array([X, Y, z], dtype=np.float32)

    def project_uv(self, p_cam):
        if p_cam is None or p_cam[2] <= 1e-9 or None in (self.fx,self.fy,self.cx,self.cy): return None
        u = self.fx * (p_cam[0] / p_cam[2]) + self.cx
        v = self.fy * (p_cam[1] / p_cam[2]) + self.cy
        return (float(u), float(v))

    # ---------- 主循環 ----------
    def loop_once(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame(); df = aligned.get_depth_frame()
            if not cf or not df: return

            if self.fx is None:
                intr = cf.get_profile().as_video_stream_profile().get_intrinsics()
                self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy
                self.get_logger().info(f"Camera intrinsics fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}")

            color = np.asanyarray(cf.get_data()); self._last_color = color
            gray_full = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            blur_full = cv2.GaussianBlur(gray_full, (self.gk,self.gk), 0)
            edge_full = cv2.Canny(blur_full, self.canny_low, self.canny_high)
            self.update_fps(); self.frame_idx += 1

            vis = color.copy()
            status = []

            # 三側處理
            status.append(self.process_side('left',  vis, gray_full, edge_full, df, (0,255,255)) or "")
            status.append(self.process_side('right', vis, gray_full, edge_full, df, (255,0,255)) or "")
            status.append(self.process_side('screw', vis, gray_full, edge_full, df, (255,255,0)) or "")

            # 三點齊 → 計算座標系 + 兩種「一個點」
            if all(self.state[k]['last_ok'] for k in ('left','right','screw')):
                vL = self.state['left']['last_xyz']
                vR = self.state['right']['last_xyz']
                vS = self.state['screw']['last_xyz']
                self.publish_tf_and_goals(vL, vR, vS)  # 也會計算 goal1/goal2

            # OSD
            y0 = 40
            for i, line in enumerate([s for s in status if s][:3]):
                cv2.putText(vis, line, (20, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, self.fs, (0,255,0), 2, cv2.LINE_AA)
            if self.show_fps:
                cv2.putText(vis, f'FPS: {self.cur_fps:.1f}', (20, y0 + 3*30),
                            cv2.FONT_HERSHEY_SIMPLEX, self.fs, (0,255,0), 2, cv2.LINE_AA)

            # 疊加顯示：兩個「一點」
            row = y0 + 4*30
            if self.goal1_cam is not None:
                gx, gy, gz = self.goal1_cam.tolist()
                cv2.putText(vis, f"GOAL1(avg3): X={gx:.3f} Y={gy:.3f} Z={gz:.3f} m",
                            (20, row), cv2.FONT_HERSHEY_SIMPLEX, self.fs, (255,255,255), 2, cv2.LINE_AA)
                if self.goal1_uv is not None:
                    gu, gv = int(round(self.goal1_uv[0])), int(round(self.goal1_uv[1]))
                    if 0 <= gu < vis.shape[1] and 0 <= gv < vis.shape[0]:
                        cv2.circle(vis, (gu, gv), 6, (255,255,255), -1)
                row += 30

            if self._save_flag:
                cv2.imwrite('lshape_onepoint_snapshot.png', vis)
                self.get_logger().info("📸 已儲存 lshape_onepoint_snapshot.png")
                self._save_flag = False

            cv2.imshow(self.win, vis)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord('q'), 27): self._should_quit = True

        except Exception as e:
            if (self.frame_idx % self.log_every_n == 0):
                self.get_logger().warning(f'循環錯誤：{e}')

        if self._should_quit:
            rclpy.shutdown()

    def process_side(self, side, vis, gray_full, edge_full, depth_frame, box_color=(0,255,255)):
        st = self.state[side]
        if st['tmpl_gray'] is None:
            st['last_ok'] = False; return None

        sx, sy, sw, sh = self.choose_search_roi(side, gray_full.shape)
        cv2.rectangle(vis, (sx, sy), (sx+sw, sy+sh), box_color, 2)

        top_left, score = self.match_template(side, gray_full, edge_full)
        label = side.upper()
        if top_left is None:
            st['miss_count'] += 1; st['last_ok'] = False
            if (self.frame_idx % self.log_every_n == 0): self.get_logger().info(f"{label}: not found")
            return None

        x_f, y_f = top_left
        th, tw = st['tmpl_gray'].shape[:2]
        x_i, y_i = int(round(x_f)), int(round(y_f))
        br = (x_i + tw, y_i + th)
        cx = x_f + float(st['tmpl_corner'][0]); cy = y_f + float(st['tmpl_corner'][1])

        # 位移限制
        jump_ok = True
        if st['prev_top_left'] is not None:
            dx = (x_f - float(st['prev_top_left'][0]))
            dy = (y_f - float(st['prev_top_left'][1]))
            jump_ok = (dx*dx + dy*dy) <= (self.max_jump * self.max_jump)

        color_box = (0,200,255) if (score >= self.thr and jump_ok) else (0,165,255)
        cv2.rectangle(vis, (x_i, y_i), br, color_box, self.lt)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 6, (0,0,255), -1)

        z = self.depth_median(depth_frame, cx, cy, 5)
        msg = f"{label}: ({int(round(cx))},{int(round(cy))})  Z={z:.3f}m  score={score:.2f}"

        if score >= self.thr_upd and jump_ok:
            st['miss_count'] = 0
            st['prev_top_left'] = (x_f, y_f)
            p = self.backproject(cx, cy, z)
            st['last_xyz'] = p if p is not None else None
            st['last_ok'] = p is not None
            if (self.frame_idx % self.pub_every_n == 0): self.info_pub.publish(String(data=msg))
            if (self.frame_idx % self.log_every_n == 0): self.get_logger().info(msg)
        else:
            st['miss_count'] += 1; st['last_ok'] = False
            if (self.frame_idx % self.log_every_n == 0):
                self.get_logger().info(msg + (" (jump too large)" if not jump_ok else " (below threshold)"))
        return msg

    # ---------- 三點 → 座標系 + 兩種「一個點」 ----------
    def publish_tf_and_goals(self, vL, vR, vS):
        """
        RIGHT 當原點；X = RIGHT→LEFT；Y = (RIGHT→SCREW) 對 X 正交化；Z = X×Y
        另外輸出：
          - GOAL1(avg3) = (vL+vR+vS)/3
          - GOAL2(projX) = vR + dot(vS-vR, x̂) * x̂    （把 SCREW 投影到 X 軸上）
        """
        # x̂
        x = vL - vR
        nx = np.linalg.norm(x)
        if nx < 1e-6: return
        x = x / nx

        # ŷ：先取 RIGHT→SCREW，再對 x̂ 正交化
        y_raw = vS - vR
        y_raw = y_raw - x * float(np.dot(y_raw, x))
        ny = np.linalg.norm(y_raw)
        if ny < 1e-6:
            y_raw = np.array([0.0,0.0,1.0], dtype=np.float32) - x * float(np.dot([0.0,0.0,1.0], x))
            ny = np.linalg.norm(y_raw)
            if ny < 1e-6: return
        y = y_raw / ny

        # ẑ
        z = np.cross(x, y); nz = np.linalg.norm(z)
        if nz < 1e-6: return
        z = z / nz

        R = np.stack([x, y, z], axis=1)  # 3x3
        t = vR.reshape(3,)

        # 目標點一：三點平均
        goal1 = (vL + vR + vS) / 3.0
        self.goal1_cam = goal1
        self.goal1_uv  = self.project_uv(goal1)

        # 目標點二：把 SCREW 投影到 X 軸（RIGHT→LEFT）
        temp_vec = vS - vR
        s = float(np.dot(temp_vec, x))          # 在 X 軸上的投影長度
        goal2 = vR + s * x
        self.goal2_cam = goal2
        self.goal2_uv  = self.project_uv(goal2)

        # 可選：仍發 TF 摘要（如不需要可註解）
        qx, qy, qz, qw = quat_from_R(R)
        msg = (f"GOAL1(avg3)=({goal1[0]:.3f},{goal1[1]:.3f},{goal1[2]:.3f})  "
               f"t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})  q=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})")
        if (self.frame_idx % self.log_every_n == 0): self.get_logger().info(msg)
        self.tf_pub.publish(String(data=msg))

    # ---------- 清理 ----------
    def destroy_node(self):
        try: self.pipeline.stop()
        except Exception: pass
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = LShapeDetectorNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
