import os
import math
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyrealsense2 as rs

# -------------------- Helpers --------------------
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

def euler_zyx_from_R(R):
    """R = [x̂ ŷ ẑ], return (yaw, pitch, roll) deg, ZYX order."""
    sy = -float(R[2,0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll  = math.atan2(float(R[2,1]), float(R[2,2]))
        yaw   = math.atan2(float(R[1,0]), float(R[0,0]))
    else:
        roll  = math.atan2(-float(R[1,2]), float(R[1,1]))
        yaw   = 0.0
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))

def quat_make_continuous(prev_q, q):
    """Avoid 180° flips between adjacent frames. q = (x,y,z,w)."""
    if prev_q is None:
        return q
    dot = prev_q[0]*q[0] + prev_q[1]*q[1] + prev_q[2]*q[2] + prev_q[3]*q[3]
    if dot < 0.0:
        return (-q[0], -q[1], -q[2], -q[3])
    return q

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

# -------------------- Node --------------------
class LShapeDetectorNode(Node):
    """
    /lshape/cmd:
      - capture_left / capture_right / capture_screw
      - reset_left   / reset_right   / reset_screw
      - save / quit
    """
    def __init__(self):
        super().__init__('lshape_detector')

        # ----- Parameters -----
        # Image & rate
        self.declare_parameter('color_width', 1280)
        self.declare_parameter('color_height', 720)
        self.declare_parameter('fps', 30)

        # Display
        self.declare_parameter('line_thickness', 3)
        self.declare_parameter('font_scale', 0.8)
        self.declare_parameter('show_fps', True)

        # Edge / template
        self.declare_parameter('canny_low', 60)
        self.declare_parameter('canny_high', 160)
        self.declare_parameter('gauss_ksize', 5)
        self.declare_parameter('tm_alpha', 0.5)
        self.declare_parameter('tm_threshold', 0.0)
        self.declare_parameter('tm_update_threshold', 0.60)
        self.declare_parameter('search_margin', 100)

        # ROI & robustness
        self.declare_parameter('max_jump_px', 80)
        self.declare_parameter('miss_limit', 10)
        self.declare_parameter('log_every_n', 10)
        self.declare_parameter('pub_every_n', 1)

        # Bootstrapping (auto request feature only when template missing / invalid)
        self.declare_parameter('auto_bootstrap', True)
        self.declare_parameter('bootstrap_period_sec', 2.0)     # per-side rate limit
        self.declare_parameter('bootstrap_on_invalid', True)     # 連續越界是否觸發外部偵測

        # 2D ROI guard（畫素）
        self.declare_parameter('roi_guard_enable', True)
        self.declare_parameter('roi_guard_margin_px', 80)

        # 幾何守門（以 R→L 基線長度為基準）
        self.declare_parameter('left_min_s_ratio', 0.20)
        self.declare_parameter('left_max_s_ratio', 1.80)
        self.declare_parameter('screw_line_dist_min_m', 0.01)
        self.declare_parameter('screw_line_dist_max_m', 0.20)
        self.declare_parameter('out_invalid_limit', 3)  # 連續 N 幀越界才觸發 re-detect

        # Pose smoothing
        self.declare_parameter('max_angle_step_deg', 3.0)
        self.declare_parameter('y_blend_beta', 0.25)

        # Depth filter toggle
        self.declare_parameter('use_depth_filters', True)

        # Template (auto) load/save
        self.declare_parameter('autoload_templates', True)

        # ----- Read params -----
        self.W = int(self.get_parameter('color_width').value)
        self.H = int(self.get_parameter('color_height').value)
        self.FPS = int(self.get_parameter('fps').value)
        self.lt = int(self.get_parameter('line_thickness').value)
        self.fs = float(self.get_parameter('font_scale').value)
        self.show_fps = bool(self.get_parameter('show_fps').value)
        self.canny_low = int(self.get_parameter('canny_low').value)
        self.canny_high = int(self.get_parameter('canny_high').value)
        self.gk = int(self.get_parameter('gauss_ksize').value); self.gk = self.gk if self.gk % 2 == 1 else 5
        self.alpha = float(self.get_parameter('tm_alpha').value)
        self.thr = float(self.get_parameter('tm_threshold').value)
        self.thr_upd = float(self.get_parameter('tm_update_threshold').value)
        self.margin = int(self.get_parameter('search_margin').value)
        self.max_jump = float(self.get_parameter('max_jump_px').value)
        self.miss_limit = int(self.get_parameter('miss_limit').value)
        self.log_every_n = int(self.get_parameter('log_every_n').value)
        self.pub_every_n = int(self.get_parameter('pub_every_n').value)

        self.auto_bootstrap = bool(self.get_parameter('auto_bootstrap').value)
        self.bootstrap_period = float(self.get_parameter('bootstrap_period_sec').value)
        self.bootstrap_on_invalid = bool(self.get_parameter('bootstrap_on_invalid').value)

        self.roi_guard_enable = bool(self.get_parameter('roi_guard_enable').value)
        self.roi_guard_margin_px = float(self.get_parameter('roi_guard_margin_px').value)
        self.left_min_s_ratio = float(self.get_parameter('left_min_s_ratio').value)
        self.left_max_s_ratio = float(self.get_parameter('left_max_s_ratio').value)
        self.screw_line_dist_min_m = float(self.get_parameter('screw_line_dist_min_m').value)
        self.screw_line_dist_max_m = float(self.get_parameter('screw_line_dist_max_m').value)
        self.out_invalid_limit = int(self.get_parameter('out_invalid_limit').value)

        self.max_angle_step_deg = float(self.get_parameter('max_angle_step_deg').value)
        self.y_blend_beta = float(self.get_parameter('y_blend_beta').value)
        self.use_depth_filters = bool(self.get_parameter('use_depth_filters').value)
        self.autoload_templates = bool(self.get_parameter('autoload_templates').value)

        # ----- ROS I/O -----
        self.info_pub = self.create_publisher(String, '/lshape_corner_info', 10)
        self.tf_pub   = self.create_publisher(String, '/lshape_tf', 10)
        self.cmd_sub  = self.create_subscription(String, '/lshape/cmd', self.on_cmd, 10)

        # external bootstrap node/topic（如需改名請同步調整）
        self.bootstrap_pub = self.create_publisher(String, '/lshape_cmd_node', 10)
        self._last_boot_ts = {'left': 0.0, 'right': 0.0, 'screw': 0.0}

        # ----- RealSense -----
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, self.FPS)
        self.config.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, self.FPS)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        self.get_logger().info('✅ RealSense started (color+depth aligned to color)')

        # Depth post-processing filters
        self.to_disp = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.holes = rs.hole_filling_filter()

        # Intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        # Window
        self.win = "L/R/Screw (X=R→L) + RIGHT XYZ"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 960, 540)

        # State
        self.last_tick = cv2.getTickCount()
        self.cur_fps = 0.0
        self.fps_ema = None
        self.fps_alpha = 0.10
        self.timer = self.create_timer(max(1.0/self.FPS, 0.01), self.loop_once)
        self._should_quit = False
        self._cleaned = False
        self._save_flag = False
        self._pause_processing = False
        self.frame_idx = 0
        self.last_ypr = None
        self.prev_quat = None
        self._last_color = None

        # Three sides (templates & tracking)
        self.state = {
            'left' : self._new_side_state(),
            'right': self._new_side_state(),
            'screw': self._new_side_state()
        }

        # Auto-load PNG templates on startup
        if self.autoload_templates:
            any_loaded = False
            for side in ('left', 'right', 'screw'):
                ok = self.load_template_from_disk(side)
                any_loaded = any_loaded or ok
            if any_loaded:
                self.get_logger().info("📂 Startup: loaded existing PNG templates (missing sides stay idle).")
            else:
                self.get_logger().info("⚠️ No PNG templates on disk. Use /lshape/cmd capture_* or external bootstrap.")

    def _new_side_state(self):
        return dict(
            tmpl_gray=None, tmpl_edge=None, tmpl_corner=None,
            prev_top_left=None, miss_count=0, last_xyz=None,
            last_ok=False, last_uv=None, last_score=0.0,
            anchor_uv=None, out_count=0
        )

    # -------------------- Template I/O --------------------
    def _tmpl_png_paths(self, side):
        return (f"template_{side}_gray.png", f"template_{side}_edge.png")

    def load_template_from_disk(self, side):
        """PNG only: template_{side}_gray.png + template_{side}_edge.png"""
        st = self.state[side]
        try:
            pg, pe = self._tmpl_png_paths(side)
            g = cv2.imread(pg, cv2.IMREAD_GRAYSCALE)
            e = cv2.imread(pe, cv2.IMREAD_GRAYSCALE)
            if g is None or e is None:
                return False

            st['tmpl_gray'] = g
            st['tmpl_edge'] = e
            if side == 'screw':
                st['tmpl_corner'] = (g.shape[1]/2.0, g.shape[0]/2.0)
            else:
                pref = 'LB' if side == 'left' else 'RB'
                u, v = harris_corner_pref(g, pref=pref)
                st['tmpl_corner'] = (float(u), float(v))

            st['prev_top_left'] = None
            st['miss_count'] = 0
            st['last_xyz'] = None
            st['last_ok'] = False
            st['last_uv'] = None
            st['anchor_uv'] = None   # 從磁碟載入時，等第一次高分匹配再鎖定
            st['out_count'] = 0

            self.get_logger().info(
                f"📂 Loaded {side.upper()} PNG (corner=({st['tmpl_corner'][0]:.2f},{st['tmpl_corner'][1]:.2f}))"
            )
            return True
        except Exception as e:
            self.get_logger().warning(f"Load PNG template failed {side}: {e}")
            return False

    def delete_template_files(self, side):
        """Remove side PNGs on disk."""
        pg, pe = self._tmpl_png_paths(side)
        for p in (pg, pe):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    # -------------------- Commands --------------------
    def on_cmd(self, msg: String):
        data = (msg.data or "").strip().lower()
        if data in ('capture_left','capture_right','capture_screw'):
            side = 'left' if 'left' in data else ('right' if 'right' in data else 'screw')
            try:
                color = self._last_color.copy()
            except Exception:
                self.get_logger().warning("No frame yet; try again later.")
                return
            self.capture_roi_as_template(color, side)
        elif data in ('reset_left','reset_right','reset_screw'):
            side = 'left' if 'left' in data else ('right' if 'right' in data else 'screw')
            self.clear_template(side)
            self.delete_template_files(side)
            self.get_logger().info(f"🗑️ Deleted {side.upper()} PNG templates from disk.")
        elif data == 'save':
            self._save_flag = True
        elif data == 'quit':
            self.request_quit()

    def request_quit(self):
        if self._cleaned:
            return
        self._should_quit = True
        try: self.timer.cancel()
        except Exception: pass
        try: self.pipeline.stop()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        self._cleaned = True
        try: rclpy.shutdown()
        except Exception: pass

    def clear_template(self, side):
        st = self.state[side]
        st.update(dict(
            tmpl_gray=None, tmpl_edge=None, tmpl_corner=None,
            prev_top_left=None, miss_count=0, last_xyz=None,
            last_ok=False, last_uv=None, last_score=0.0,
            anchor_uv=None, out_count=0
        ))
        self.get_logger().info(f"🧹 Cleared {side.upper()} template (RAM).")

    def capture_roi_as_template(self, color_bgr, side):
        # 暫停主循環，避免 ROI 選取時卡住
        self._pause_processing = True
        try:
            title = f"Select {side. upper()} ROI (Enter/ESC)"
            box = cv2.selectROI(title, color_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(title)
            x,y,w,h = [int(v) for v in box]
            if w<=0 or h<=0:
                self.get_logger().warning("No ROI selected.")
                return
            patch = color_bgr[y:y+h, x:x+w]
            gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            blur  = cv2.GaussianBlur(gray, (self.gk,self.gk), 0)
            edge  = cv2.Canny(blur, self.canny_low, self.canny_high)

            if side == 'screw':
                u_refined = w/2.0; v_refined = h/2.0
            else:
                pref = 'LB' if side=='left' else 'RB'
                u0, v0 = harris_corner_pref(gray, pref=pref)
                corners = np.array([[float(u0), float(v0)]], dtype=np.float32).reshape(-1,1,2)
                try:
                    cv2.cornerSubPix(gray, corners, (7,7), (-1,-1),
                                     (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-2))
                    u_refined = float(corners[0,0,0]); v_refined = float(corners[0,0,1])
                except Exception:
                    u_refined, v_refined = float(u0), float(v0)

            st = self.state[side]
            st['tmpl_gray'] = gray
            st['tmpl_edge'] = edge
            st['tmpl_corner'] = (u_refined, v_refined)
            st['prev_top_left'] = (x, y)
            st['miss_count'] = 0
            st['last_xyz'] = None
            st['last_ok'] = False
            st['last_uv'] = None
            # 直接以目前角點/中心「全域像素座標」當錨點
            st['anchor_uv'] = (x + float(u_refined), y + float(v_refined))
            st['out_count'] = 0

            pg, pe = self._tmpl_png_paths(side)
            cv2.imwrite(pg, gray)
            cv2.imwrite(pe, edge)
            self.get_logger().info(f"🎯 {side.upper()} template: rect=({x},{y},{w},{h}) corner=({u_refined:.2f},{v_refined:.2f})")
        finally:
            self._pause_processing = False

    # -------------------- Utils --------------------
    def update_fps(self):
        now = cv2.getTickCount()
        dt  = (now - self.last_tick) / cv2.getTickFrequency()
        self.last_tick = now
        if dt <= 0: return
        inst = 1.0 / dt
        if self.fps_ema is None:
            self.fps_ema = inst
        else:
            self.fps_ema = (1.0 - self.fps_alpha) * self.fps_ema + self.fps_alpha * inst
        self.cur_fps = self.fps_ema

    def depth_median(self, depth_frame, u, v, win=7):
        try:
            _ = depth_frame.get_distance(0, 0)
        except Exception:
            depth_frame = depth_frame.as_depth_frame()
            try:
                _ = depth_frame.get_distance(0, 0)
            except Exception:
                return 0.0

        h = depth_frame.get_height();  w = depth_frame.get_width()
        half = max(1, win // 2)
        uu, vv = int(round(u)), int(round(v))
        xs = range(max(0, uu-half), min(w, uu+half+1))
        ys = range(max(0, vv-half), min(h, vv+half+1))
        vals = []
        for yy in ys:
            for xx in xs:
                d = depth_frame.get_distance(xx, yy)
                if d > 0:
                    vals.append(d)
        return float(np.median(vals)) if vals else 0.0

    def choose_search_roi(self, side, full_shape):
        st = self.state[side];  H, W = full_shape[:2]
        if st['tmpl_gray'] is None or st['prev_top_left'] is None or st['miss_count'] >= self.miss_limit:
            return (0,0,W,H)
        x, y = st['prev_top_left'];  th, tw = st['tmpl_gray'].shape[:2]
        xm = max(0, int(round(x)) - self.margin)
        ym = max(0, int(round(y)) - self.margin)
        xM = min(W, int(round(x)) + tw + self.margin)
        yM = min(H, int(round(y)) + th + self.margin)
        return (xm, ym, xM - xm, yM - ym)

    def match_template(self, side, gray_full, edge_full):
        st = self.state[side];  th, tw = st['tmpl_gray'].shape[:2]
        sx, sy, sw, sh = self.choose_search_roi(side, gray_full.shape)
        search_g = gray_full[sy:sy+sh, sx:sx+sw]
        search_e = edge_full[sy:sy+sh, sx:sx+sw]
        if search_g.shape[0] < th or search_g.shape[1] < tw:
            return None, None
        res_g = cv2.matchTemplate(search_g, st['tmpl_gray'], cv2.TM_CCOEFF_NORMED)
        res_e = cv2.matchTemplate(search_e, st['tmpl_edge'], cv2.TM_CCOEFF_NORMED)
        res = self.alpha * res_g + (1.0 - self.alpha) * res_e
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        subx, suby = refine_peak(res, max_loc)
        return (sx + subx, sy + suby), float(max_val)

    def backproject(self, u, v, z):
        if None in (self.fx, self.fy, self.cx, self.cy) or z <= 0:
            return None
        X = (u - self.cx) / self.fx * z
        Y = (v - self.cy) / self.fy * z
        return np.array([X, Y, z], dtype=np.float32)

    def project_uv(self, p_cam):
        if p_cam is None or p_cam[2] <= 1e-9 or None in (self.fx,self.fy,self.cx,self.cy):
            return None
        u = self.fx * (p_cam[0] / p_cam[2]) + self.cx
        v = self.fy * (p_cam[1] / p_cam[2]) + self.cy
        return (float(u), float(v))

    @staticmethod
    def _wrap_deg(a):
        return (a + 180.0) % 360.0 - 180.0

    def _smooth_ypr(self, y, p, r):
        curr = np.array([y, p, r], dtype=np.float32)
        if self.last_ypr is None:
            return (float(self._wrap_deg(curr[0])),
                    float(self._wrap_deg(curr[1])),
                    float(self._wrap_deg(curr[2])))
        prev = np.array(self.last_ypr, dtype=np.float32)
        diff = curr - prev
        diff[0] = self._wrap_deg(diff[0])
        diff[1] = self._wrap_deg(diff[1])
        diff[2] = self._wrap_deg(diff[2])
        m = self.max_angle_step_deg
        step = np.clip(diff, -m, m)
        smoothed = prev + step
        smoothed[0] = self._wrap_deg(smoothed[0])
        smoothed[1] = self._wrap_deg(smoothed[1])
        smoothed[2] = self._wrap_deg(smoothed[2])
        return float(smoothed[0]), float(smoothed[1]), float(smoothed[2])

    # -------------------- Auto bootstrap --------------------
    def auto_bootstrap_templates(self):
        """Only when template is missing: ask an external node to detect features."""
        if not self.auto_bootstrap:
            return
        now = time.time()
        for side in ('left','right','screw'):
            st = self.state[side]
            if st['tmpl_gray'] is None:
                if now - self._last_boot_ts[side] >= self.bootstrap_period and not self._pause_processing:
                    self._bootstrap_send(side)
                    self._last_boot_ts[side] = now

    def _bootstrap_send(self, side):
        cmd = f"auto_detect_{side}"
        self.bootstrap_pub.publish(String(data=cmd))
        self.get_logger().info(f"📣 Requesting external feature detect: {cmd}")

    # -------------------- Validation & reporting --------------------
    def validate_frame(self, vL, vR, vS):
        issues = []
        if vL is None or vR is None or vS is None:
            issues.append('missing_points')
            return False, issues

        x = vL - vR
        nx = float(np.linalg.norm(x))
        if nx <= 1e-6:
            issues.append('baseline_zero')
            return False, issues
        xhat = x / nx

        # LEFT 在 X 軸上的投影需落在比例範圍
        sL = float(np.dot(vL - vR, xhat))
        if not (self.left_min_s_ratio * nx <= sL <= self.left_max_s_ratio * nx):
            issues.append('left_out_of_x_range')

        # SCREW 與 X 軸的垂距需落在走廊
        sS = float(np.dot(vS - vR, xhat))
        dS = float(np.linalg.norm((vS - vR) - sS * xhat))
        if not (self.screw_line_dist_min_m <= dS <= self.screw_line_dist_max_m):
            issues.append('screw_out_of_corridor')

        return (len(issues) == 0), issues

    def report_invalid_and_maybe_bootstrap(self, issues):
        msg = "[INVALID] reasons=" + ",".join(issues)
        if (self.frame_idx % self.log_every_n == 0):
            self.get_logger().info(msg)
        if not self.bootstrap_on_invalid:
            return
        # 針對性 re-detect
        if 'left_out_of_x_range' in issues:
            self._bootstrap_send('left')
        if 'screw_out_of_corridor' in issues:
            self._bootstrap_send('screw')
        if 'baseline_zero' in issues or 'missing_points' in issues:
            for s in ('left','right','screw'):
                self._bootstrap_send(s)

    # -------------------- Main loop --------------------
    def loop_once(self):
        try:
            if self._pause_processing:
                return

            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame();  df = aligned.get_depth_frame()
            if not cf or not df:
                return

            # keep depth as depth_frame
            df = df.as_depth_frame()
            if not df:
                return

            # Depth filters (optional)
            if self.use_depth_filters:
                dfp = self.to_disp.process(df)
                dfp = self.spatial.process(dfp)
                dfp = self.temporal.process(dfp)
                dfp = self.holes.process(dfp)
                df  = self.to_depth.process(dfp)

            if self.fx is None:
                intr = cf.get_profile().as_video_stream_profile().get_intrinsics()
                self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy
                self.get_logger().info(f"Intrinsics fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}")

            color = np.asanyarray(cf.get_data()); self._last_color = color
            gray_full = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            blur_full = cv2.GaussianBlur(gray_full, (self.gk,self.gk), 0)
            edge_full = cv2.Canny(blur_full, self.canny_low, self.canny_high)
            self.update_fps(); self.frame_idx += 1

            # 啟動時/缺模板時才請外部偵測
            self.auto_bootstrap_templates()

            vis = color.copy()

            # Run three sides
            status_left  = self.process_side('left',  vis, gray_full, edge_full, df, (0,255,255))
            status_right = self.process_side('right', vis, gray_full, edge_full, df, (255,0,255))
            status_screw = self.process_side('screw', vis, gray_full, edge_full, df, (255,255,0))

            # Publish/log three lines (throttled)
            for msg in (status_left, status_right, status_screw):
                if msg:
                    if (self.frame_idx % self.pub_every_n == 0):
                        self.info_pub.publish(String(data=msg))
                    if (self.frame_idx % self.log_every_n == 0):
                        self.get_logger().info(msg)

            # If all three valid → publish TF (with validation & smoothing)
            if all(self.state[k]['last_ok'] for k in ('left','right','screw')):
                self.publish_tf(self.state['left']['last_xyz'],
                                self.state['right']['last_xyz'],
                                self.state['screw']['last_xyz'])

            # OSD text
            y0 = 40
            for i, line in enumerate([s for s in (status_left, status_right, status_screw) if s][:3]):
                cv2.putText(vis, line, (20, y0 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, self.fs, (0,255,0), 2, cv2.LINE_AA)

            # RIGHT xyz
            ry = self.state['right']['last_xyz']
            if ry is not None:
                X,Y,Z = ry.tolist()
                cv2.putText(vis, f"RIGHT (cam): X={X:.3f}  Y={Y:.3f}  Z={Z:.3f} m",
                            (20, y0 + 3*30), cv2.FONT_HERSHEY_SIMPLEX, self.fs, (255,255,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, "RIGHT (cam): ---",
                            (20, y0 + 3*30), cv2.FONT_HERSHEY_SIMPLEX, self.fs, (255,255,255), 2, cv2.LINE_AA)

            # YPR
            if self.last_ypr is not None:
                yaw_deg, pitch_deg, roll_deg = self.last_ypr
                cv2.putText(vis, f"YPR(deg): Y={yaw_deg:.2f}  P={pitch_deg:.2f}  R={roll_deg:.2f}",
                            (20, y0 + 5*30), cv2.FONT_HERSHEY_SIMPLEX, self.fs, (0,255,255), 2, cv2.LINE_AA)

            # FPS
            if self.show_fps:
                cv2.putText(vis, f'FPS: {self.cur_fps:.1f}', (20, y0 + 4*30),
                            cv2.FONT_HERSHEY_SIMPLEX, self.fs, (0,255,0), 2, cv2.LINE_AA)

            # Save?
            if self._save_flag:
                cv2.imwrite('lshape_snapshot.png', vis)
                self.get_logger().info("📸 Saved lshape_snapshot.png")
                self._save_flag = False

            # Show & quit key
            cv2.imshow(self.win, vis)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord('q'), 27):
                self.request_quit()

        except Exception as e:
            if (self.frame_idx % self.log_every_n == 0):
                self.get_logger().warning(f'Loop error: {e}')

        if self._should_quit and not self._cleaned:
            self.request_quit()

    # -------------------- Per-side processing --------------------
    def process_side(self, side, vis, gray_full, edge_full, depth_frame, box_color=(0,255,255)):
        st = self.state[side]
        if st['tmpl_gray'] is None:
            st['last_ok'] = False
            return f"{side.upper()}: template missing"

        sx, sy, sw, sh = self.choose_search_roi(side, gray_full.shape)
        cv2.rectangle(vis, (sx, sy), (sx+sw, sy+sh), box_color, 2)

        top_left, score = self.match_template(side, gray_full, edge_full)
        label = side.upper()
        if top_left is None:
            st['miss_count'] += 1; st['last_ok'] = False
            st['last_score'] = 0.0
            return f"{label}: not found"

        x_f, y_f = top_left
        th, tw = st['tmpl_gray'].shape[:2]
        x_i, y_i = int(round(x_f)), int(round(y_f))
        br = (x_i + tw, y_i + th)
        cx = x_f + float(st['tmpl_corner'][0])
        cy = y_f + float(st['tmpl_corner'][1])

        # 若從磁碟載入沒有錨點→第一次高分匹配時鎖定錨點
        if st['anchor_uv'] is None and score >= max(self.thr_upd, 0.65):
            st['anchor_uv'] = (float(cx), float(cy))
            st['out_count'] = 0

        # Jump guard
        jump_ok = True
        if st['prev_top_left'] is not None:
            dx = (x_f - float(st['prev_top_left'][0]))
            dy = (y_f - float(st['prev_top_left'][1]))
            jump_ok = (dx*dx + dy*dy) <= (self.max_jump * self.max_jump)

        color_box = (0,200,255) if (score >= self.thr and jump_ok) else (0,165,255)
        cv2.rectangle(vis, (x_i, y_i), br, color_box, self.lt)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 6, (0,0,255), -1)

        # 2D ROI 守門：離錨點過遠，直接回報越界
        if self.roi_guard_enable and st['anchor_uv'] is not None:
            ax, ay = st['anchor_uv']
            if abs(cx - ax) > self.roi_guard_margin_px or abs(cy - ay) > self.roi_guard_margin_px:
                st['last_ok'] = False
                st['last_score'] = float(score)
                st['out_count'] += 1
                if self.bootstrap_on_invalid and st['out_count'] >= self.out_invalid_limit:
                    self._bootstrap_send(side)
                    st['out_count'] = 0
                return f"{label}: out_of_target_area (roi)"
            else:
                st['out_count'] = 0

        z = self.depth_median(depth_frame, cx, cy, 7)
        if z <= 0:
            st['miss_count'] += 1; st['last_ok'] = False
            st['last_score'] = float(score)
            return f"{label}: ({int(round(cx))},{int(round(cy))})  Z=0.000m  score={score:.2f} (invalid depth)"

        msg = f"{label}: ({int(round(cx))},{int(round(cy))})  Z={z:.3f}m  score={score:.2f}"

        if score >= self.thr_upd and jump_ok:
            st['miss_count'] = 0
            st['prev_top_left'] = (x_f, y_f)
            p = self.backproject(cx, cy, z)
            st['last_xyz'] = p if p is not None else None
            st['last_ok'] = p is not None
            st['last_uv'] = (float(cx), float(cy))
            st['last_score'] = float(score)
        else:
            st['miss_count'] += 1
            st['last_ok'] = False
            st['last_score'] = float(score)
        return msg

    # -------------------- 3 points → TF (YPR stabilized) --------------------
    def publish_tf(self, vL, vR, vS):
        ok, issues = self.validate_frame(vL, vR, vS)
        if not ok:
            self.report_invalid_and_maybe_bootstrap(issues)
            return

        # x̂ = RIGHT→LEFT
        x = vL - vR
        nx = np.linalg.norm(x)
        if nx < 1e-6: return
        x = x / nx

        # y_raw = RIGHT→SCREW, remove X component
        y_raw = vS - vR
        y_raw = y_raw - x * float(np.dot(y_raw, x))

        # blend toward camera Z projected to ⟂X plane for roll stability
        z_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        y_pref = z_cam - x * float(np.dot(z_cam, x))
        y_raw = (1.0 - self.y_blend_beta) * y_raw + self.y_blend_beta * y_pref

        ny = np.linalg.norm(y_raw)
        if ny < 1e-6: return
        y = y_raw / ny

        # ẑ and re-orthogonalize
        z = np.cross(x, y); nz = np.linalg.norm(z)
        if nz < 1e-6: return
        z = z / nz
        y = np.cross(z, x); y = y / np.linalg.norm(y)

        Rm = np.column_stack((x, y, z))
        qx, qy, qz, qw = quat_from_R(Rm)
        q = quat_make_continuous(self.prev_quat, (qx,qy,qz,qw))
        self.prev_quat = q

        yaw_deg, pitch_deg, roll_deg = euler_zyx_from_R(Rm)
        yaw_deg, pitch_deg, roll_deg = self._smooth_ypr(yaw_deg, pitch_deg, roll_deg)
        self.last_ypr = (yaw_deg, pitch_deg, roll_deg)

        msg = f"ypr_deg=({yaw_deg:.2f},{pitch_deg:.2f},{roll_deg:.2f})"
        self.tf_pub.publish(String(data=msg))
        if (self.frame_idx % self.log_every_n == 0):
            self.get_logger().info(msg)

    # -------------------- Cleanup --------------------
    def destroy_node(self):
        try:
            if not self._cleaned:
                self.pipeline.stop()
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

# -------------------- Main --------------------
def main():
    rclpy.init()
    node = LShapeDetectorNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        node.request_quit()
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
