"""PLY 파일 저장 유틸리티 (3DGS 포맷)"""

import numpy as np
from plyfile import PlyData, PlyElement


def save_ply_with_normals(pts, colors, path):
    """
    3DGS가 요구하는 포맷으로 PLY 저장.
    (x, y, z, nx, ny, nz, red, green, blue)

    Args:
        pts   : (N, 3) float32 3D points
        colors: (N, 3) uint8 RGB colors
        path  : output .ply path
    """
    mask    = np.isfinite(pts).all(axis=1)
    pts     = pts[mask].astype(np.float32)
    colors  = colors[mask].astype(np.uint8)
    normals = np.zeros_like(pts)

    verts = np.array(
        [(p[0], p[1], p[2],
          n[0], n[1], n[2],
          c[0], c[1], c[2])
         for p, n, c in zip(pts, normals, colors)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    PlyData([PlyElement.describe(verts, 'vertex')]).write(path)
    return len(pts)
