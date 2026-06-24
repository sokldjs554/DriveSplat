# DriveSplat

**Depth-Guided 3D Gaussian Splatting Initialization**

단안 카메라 영상에서 Depth Anything V2로 추정한 depth map을  
3D Gaussian Splatting의 초기화에 활용하는 파이프라인입니다.

---

## 실험 결과 (Ablation Study)

| Method | Init Points | PSNR ↑ |
|--------|------------|--------|
| A) COLMAP only (baseline) | 136,029 | **25.15 dB** |
| B) Depth only | 656,224 | 18.89 dB |
| C) COLMAP + Depth (per-frame scale) | 792,253 | 18.85 dB |
| D) COLMAP + Depth (global scale) | 1,477,895 | ~19.5 dB |

### 주요 발견
- COLMAP sparse 초기화(A)가 정적 씬에서 baseline으로 강력함
- Depth-guided 방법(B~D)은 **relative depth의 scale 분산**(σ=0.27)이  
  포인트 품질을 저하시켜 PSNR이 낮아짐
- 이는 **metric depth fine-tuning의 필요성**을 실험으로 직접 증명

---

## 직접 구현한 것 (★)

| 파일 | 내용 |
|------|------|
| `core/scale_alignment.py` | COLMAP anchor 기반 scale 추정 (median ratio) |
| `core/depth_to_pointcloud.py` | Backprojection 역투영 구현 |
| `core/metrics.py` | KITTI depth 평가 지표, PSNR 계산 |
| `core/ply_utils.py` | 3DGS 포맷 PLY 저장 |

---

## 시스템 구조
---

## 설치

```bash
git clone https://github.com/sokldjs554/DriveSplat
cd DriveSplat
pip install -r requirements.txt
```

---

## 데이터셋

- [3DGS Truck Scene](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [KITTI Depth Completion](http://www.cvlibs.net/datasets/kitti/)

---

## 환경

- Python 3.12
- PyTorch 2.x
- CUDA 12.x
- Google Colab (T4 GPU)

---

## 참고 논문

- Kerbl et al., "3D Gaussian Splatting", SIGGRAPH 2023
- Yang et al., "Depth Anything V2", NeurIPS 2024

## 실험 결과 시각화

![Ablation Study](assets/results/ablation_final.png)

![Point Cloud BEV](assets/results/bev_comparison.png)
