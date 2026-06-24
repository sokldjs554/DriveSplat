"""
COLMAP Anchor-based Depth Scale Alignment
★ Custom implementation

핵심 아이디어:
    Depth Anything V2의 relative depth를 COLMAP sparse points를
    anchor로 활용하여 metric scale로 변환합니다.

    d_metric = scale * d_relative
    scale = median(d_colmap / d_pred) at anchor pixels
"""

import numpy as np


def project_colmap_to_image(colmap_pts, K, R_c2w, pos_world, H, W):
    """
    COLMAP 3D points를 이미지 평면에 투영합니다.

    Args:
        colmap_pts: (N, 3) COLMAP sparse 3D points
        K         : (3, 3) camera intrinsics
        R_c2w     : (3, 3) camera-to-world rotation
        pos_world : (3,)   camera world position
        H, W      : image height, width
    Returns:
        u, v      : projected pixel coordinates
        d_col     : depth in camera space (meters)
    """
    R_w2c   = R_c2w.T
    t_w2c   = -R_w2c @ pos_world
    pts_cam = (R_w2c @ colmap_pts.T).T + t_w2c

    front   = pts_cam[:, 2] > 0.001
    pts_cam = pts_cam[front]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u     = (pts_cam[:, 0] / pts_cam[:, 2] * fx + cx).astype(int)
    v     = (pts_cam[:, 1] / pts_cam[:, 2] * fy + cy).astype(int)
    d_col = pts_cam[:, 2]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[in_img], v[in_img], d_col[in_img]


def estimate_scale_per_frame(d_pred_map, u, v, d_colmap,
                              orig_H, orig_W, min_anchors=10):
    """
    단일 프레임에서 COLMAP anchor 기반 scale 추정.

    Args:
        d_pred_map: (dH, dW) Depth Anything V2 출력
        u, v      : COLMAP anchor 픽셀 좌표 (원본 해상도)
        d_colmap  : anchor의 실제 depth (meters)
        orig_H/W  : 원본 이미지 해상도
        min_anchors: 최소 anchor 수
    Returns:
        scale (float) or None
    """
    dH, dW = d_pred_map.shape

    # 원본 → depth map 좌표 변환
    u_s = np.clip((u * dW / orig_W).astype(int), 0, dW - 1)
    v_s = np.clip((v * dH / orig_H).astype(int), 0, dH - 1)

    d_pred = d_pred_map[v_s, u_s]
    valid  = d_pred > 0.01

    if valid.sum() < min_anchors:
        return None

    # Median ratio: outlier에 robust
    scale = np.median(d_colmap[valid] / d_pred[valid])
    return float(scale) if scale > 0 else None


def estimate_global_scale(depth_model, img_paths, cam_dict,
                           colmap_pts, device='cuda'):
    """
    전체 프레임에서 통합 global scale 추정.
    프레임별 scale variance를 줄여 일관성 확보.

    Args:
        depth_model: Depth Anything V2 모델
        img_paths  : 이미지 경로 리스트
        cam_dict   : cameras.json 딕셔너리
        colmap_pts : (N, 3) COLMAP 3D points
    Returns:
        global_scale (float)
    """
    import cv2, torch, os

    all_d_col  = []
    all_d_pred = []

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        if img_name not in cam_dict:
            continue

        ci     = cam_dict[img_name]
        orig_H = ci['height']
        orig_W = ci['width']
        K      = np.array([[ci['fx'], 0, orig_W / 2],
                            [0, ci['fy'], orig_H / 2],
                            [0, 0, 1]])
        R_c2w     = np.array(ci['rotation'])
        pos_world = np.array(ci['position'])

        raw = cv2.cvtColor(
            cv2.imread(img_path), cv2.COLOR_BGR2RGB
        )
        with torch.no_grad():
            d_pred = depth_model.infer_image(raw)
        dH, dW = d_pred.shape

        u, v, d_col = project_colmap_to_image(
            colmap_pts, K, R_c2w, pos_world, orig_H, orig_W
        )
        if len(u) < 10:
            continue

        u_s   = np.clip((u * dW / orig_W).astype(int), 0, dW-1)
        v_s   = np.clip((v * dH / orig_H).astype(int), 0, dH-1)
        d_anc = d_pred[v_s, u_s]
        valid = d_anc > 0.01

        all_d_col.append(d_col[valid])
        all_d_pred.append(d_anc[valid])

    all_d_col  = np.concatenate(all_d_col)
    all_d_pred = np.concatenate(all_d_pred)
    return float(np.median(all_d_col / all_d_pred))
