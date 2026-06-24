"""
Depth Map → 3D Point Cloud (Backprojection)
★ Custom implementation

핵심 수식:
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    pts_world = R_c2w @ pts_cam + pos_world
"""

import numpy as np


def backproject_depth(depth_metric, fx, fy, cx, cy,
                      R_c2w, pos_world,
                      scene_center, scene_radius,
                      sample_rate=4):
    """
    Metric depth map을 3D world point cloud로 변환합니다.

    Args:
        depth_metric : (H, W) metric depth map (meters)
        fx, fy       : focal lengths
        cx, cy       : principal point
        R_c2w        : (3, 3) camera-to-world rotation
        pos_world    : (3,)   camera world position
        scene_center : (3,)   scene center for filtering
        scene_radius : float  max distance from scene center
        sample_rate  : pixel downsampling rate
    Returns:
        pts_world : (N, 3) 3D points in world coordinates
        valid_mask: (M,) boolean mask
    """
    H, W = depth_metric.shape

    # depth map 기준 camera intrinsics 스케일 조정
    fx_d = fx * W / (W * 1.0)  # depth map과 원본 동일 시 1.0
    fy_d = fy * H / (H * 1.0)
    cx_d = cx
    cy_d = cy

    # 픽셀 그리드 생성
    u_grid, v_grid = np.meshgrid(
        np.arange(0, W, sample_rate),
        np.arange(0, H, sample_rate)
    )
    u = u_grid.flatten().astype(np.float64)
    v = v_grid.flatten().astype(np.float64)
    d = depth_metric[::sample_rate, ::sample_rate].flatten()
    d = d.astype(np.float64)

    # ★ 역투영: 이미지 좌표 → 카메라 좌표
    x_cam = (u - cx_d) * d / fx_d
    y_cam = (v - cy_d) * d / fy_d
    z_cam = d
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # ★ 카메라 → 월드 좌표 변환
    pts_world = (R_c2w @ pts_cam.T).T + pos_world

    # 씬 범위 필터링 (outlier 제거)
    dist      = np.linalg.norm(pts_world - scene_center, axis=1)
    valid     = (dist < scene_radius) & (d > 0.1)

    return pts_world[valid], valid


def build_pointcloud(depth_model, img_paths, cam_dict,
                     colmap_pts, scene_center, scene_radius,
                     global_scale, sample_every=10,
                     pixel_sample=4, device='cuda'):
    """
    여러 뷰의 depth map으로 dense point cloud 생성.

    Args:
        global_scale: estimate_global_scale()로 추정한 값
    Returns:
        all_pts   : (N, 3) merged point cloud
        all_colors: (N, 3) RGB colors
    """
    import cv2, torch, os
    from PIL import Image

    selected  = img_paths[::sample_every]
    all_pts   = []
    all_colors = []

    for img_path in selected:
        img_name = os.path.basename(img_path)
        if img_name not in cam_dict:
            continue

        ci        = cam_dict[img_name]
        orig_H    = ci['height']
        orig_W    = ci['width']
        fx, fy    = ci['fx'], ci['fy']
        cx_cam    = orig_W / 2.0
        cy_cam    = orig_H / 2.0
        R_c2w     = np.array(ci['rotation'])
        pos_world = np.array(ci['position'])

        raw = cv2.cvtColor(
            cv2.imread(img_path), cv2.COLOR_BGR2RGB
        )
        with torch.no_grad():
            d_pred = depth_model.infer_image(raw)
        dH, dW = d_pred.shape

        # depth map 해상도에 맞게 intrinsics 조정
        fx_d = fx * dW / orig_W
        fy_d = fy * dH / orig_H
        cx_d = cx_cam * dW / orig_W
        cy_d = cy_cam * dH / orig_H

        # Metric depth 변환
        d_metric = np.clip(
            d_pred.astype(np.float64) * global_scale,
            0.1, scene_radius * 1.5
        )

        pts, valid = backproject_depth(
            d_metric, fx_d, fy_d, cx_d, cy_d,
            R_c2w, pos_world,
            scene_center, scene_radius * 0.6,
            sample_rate=pixel_sample
        )

        if len(pts) == 0:
            continue

        # 색상 추출
        img_arr = np.array(Image.open(img_path))
        img_rsz = cv2.resize(img_arr, (dW, dH))
        c_all   = img_rsz[::pixel_sample,
                           ::pixel_sample].reshape(-1, 3)
        n       = min(len(valid), len(c_all))
        colors  = c_all[:n][valid[:n]]
        pts     = pts[:len(colors)]

        all_pts.append(pts)
        all_colors.append(colors)

    return (np.concatenate(all_pts, axis=0),
            np.concatenate(all_colors, axis=0))
