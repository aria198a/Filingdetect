import os
import numpy as np
import open3d as o3d

class ICPFITTER:
    def __init__(self, golden_dir=None, default_region="screw", use_icp=True):
        if golden_dir is None:
            golden_dir = os.path.abspath(os.path.join(os.getcwd(), "golden_samples"))
        os.makedirs(golden_dir, exist_ok=True)

        self.golden_dir = golden_dir
        self.paths = {
            "screw":   os.path.join(golden_dir, "golden_screw.pcd"),
            "battery": os.path.join(golden_dir, "golden_battery.pcd"),
        }
        self.default_region = default_region
        self.use_icp = use_icp

        
        self.golden = {k: (o3d.io.read_point_cloud(p) if os.path.exists(p) else None)
                       for k, p in self.paths.items()}
        for k, p in self.paths.items():
            if self.golden[k] is not None:
                print(f"✅ 載入 {k} golden: {p} (點數={len(self.golden[k].points)})")

    # ---- 對外 API -----------------------------------------------------------

    def save_golden(self, color_image, depth_image, depth_intrin, region="screw"):
        """用當下影像建立點雲並存成該區域的 golden 檔"""
        pcd = self.rs_to_pointcloud(color_image, depth_image, depth_intrin)
        if pcd is None or len(pcd.points) == 0:
            print("⚠️ 無法從當前影像建立點雲，未儲存 golden")
            return False

        path = self._region_path(region)
        ok = o3d.io.write_point_cloud(path, pcd)
        if ok:
            self.golden[region] = pcd
            print(f"✅ 已儲存 {region} golden: {path} (點數={len(pcd.points)})")
            return True
        else:
            print(f"❌ 儲存失敗: {path}")
            return False

    def icp_fit(self, color_image, depth_image, depth_intrin, region=None):

        region = (region or self.default_region).lower()
        g = self.golden.get(region, None)
        if g is None:
            print(f"⚠️ {region} golden 不存在，請先 save {region}")
            return None

        cur = self.rs_to_pointcloud(color_image, depth_image, depth_intrin)
        if cur is None or len(cur.points) == 0:
            print("⚠️ 無法建立當前點雲")
            return None

        # 下採樣與法線（穩定 ICP）
        g_d = g.voxel_down_sample(0.003)     # 3mm
        c_d = cur.voxel_down_sample(0.003)
        g_d.estimate_normals()
        c_d.estimate_normals()

        if self.use_icp:
            # 距離門檻視場景而定：2~5cm 常見
            threshold = 0.05
            init = np.eye(4)
            result = o3d.pipelines.registration.registration_icp(
                c_d, g_d, threshold, init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            # ICP 對齊後的 inlier RMSE
            return float(result.inlier_rmse)
        else:
            # 不對齊，直接最近點平均距離（姿態不同時會很大）
            d = np.mean(g_d.compute_point_cloud_distance(c_d))
            return float(d)

    # ---- 內部工具 -----------------------------------------------------------

    def _region_path(self, region):
        region = region.lower()
        if region not in self.paths:
            raise ValueError(f"未知區域: {region}，應為 'screw' 或 'battery'")
        return self.paths[region]

    def rs_to_pointcloud(self, color, depth, intrin):
        """
        color: HxWx3 uint8 (BGR/RGB 皆可)
        depth: HxW uint16 (深度 raw)
        intrin: RealSense intrinsics 物件 (含 fx, fy, ppx, ppy)
        """
        if color is None or depth is None or intrin is None:
            return None
        h, w = depth.shape
        fx, fy = intrin.fx, intrin.fy
        cx, cy = intrin.ppx, intrin.ppy

        # 依你的裝置調整（D435 常見 0.001）
        depth_scale = 0.001
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32) * depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = color.reshape(-1, 3).astype(np.float32) / 255.0

        valid = (z.reshape(-1) > 0) & (z.reshape(-1) < 1.5)
        points = points[valid]
        colors = colors[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
