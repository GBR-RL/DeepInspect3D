![DeepInspect3D Banner](https://raw.githubusercontent.com/gbr-rl/DeepInspect3D/main/docs/banner.png)

# DeepInspect3D - 3D Mesh Inspection with Open3D-Based Point Cloud Alignment, Defect Localization, and Metric Evaluation

![Computer Vision](https://img.shields.io/badge/Project-Computer%20Vision-blue)
![Image Processing](https://img.shields.io/badge/Project-Image%20Processing-yellow)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5-green)
[![Made with Blender](https://img.shields.io/badge/rendered%20with-blender-orange?logo=blender)](https://www.blender.org/)
[![Open3D](https://img.shields.io/badge/Built%20with-Open3D-1f425f.svg?logo=Open3D&logoColor=white)](http://www.open3d.org/)


**DeepInspect3D** delivers an extensible 3D inspection pipeline for high-precision geometric defect detection in mechanical components. It integrates photorealistic synthetic rendering using Blender, point cloud generation, multi-stage filtering, FPFH+ICP alignment, and deviation field analysis against CAD ground truth, enabling automated metrology-grade evaluation and quality assurance.

---

## Project Structure Overview

```plaintext
DeepInspect3D/
├── notebook/
│   └── inspection_pipeline.ipynb      # Unified inspection notebook
├── scripts/                          # Modular inspection stages
│   ├── render.py                     # Blender-based RGB/Depth/Normal rendering
│   ├── pointcloud.py                 # STL → PLY conversion
│   ├── preprocess_pointcloud.py      # Denoising, voxelization, normals
│   ├── register_pointcloud.py        # Global (FPFH) + ICP registration
│   ├── detect_defects.py             # Deviation analysis + defect mask
│   └── evaluation_metrics.py         # Chamfer, deviation, precision/recall
├── data/
│   ├── reference_master.stl          # CAD ground truth model
│   ├── test_scanned.stl              # QA scanned model
│   └── processed/
│       ├── reference_clean.ply
│       ├── test_clean.ply
│       └── test_aligned.ply
├── defects/
│   ├── deviation_heatmap.ply         # Color-mapped geometric deviation
│   └── binary_defect_mask.ply        # Thresholded defective regions
├── reports/
│   ├── metrics.json                  # Raw inspection metrics
│   └── inspection_report.json        # Final summary
├── blender_assets/                   # HDRI, textures, config
│   ├── textures/
│   ├── hdri/
│   └── config.yaml
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation and usage
```

---

## Key Features

- **CAD-to-Scan Inspection**  
  Compare scanned parts against their CAD references for precise geometry verification.

- **Modular Point Cloud Pipeline**  
  Includes denoising, downsampling, normal estimation, global and local registration.

- **Deviation Mapping + Masking**  
  Visual and binary extraction of out-of-tolerance regions using thresholded spatial error maps.

- **Synthetic Data Support**  
  Procedural rendering of RGB, depth, and normal maps from `.stl` with Blender (HDRI + PBR materials).

- **Quantitative Evaluation**  
  Metrics include:
  - Mean & Max Surface Deviation
  - Chamfer Distance (bidirectional)
  - Precision / Recall / F1 (if ground truth available)

---

## 📊 Outputs

| Artifact                 | Path                                      |
|--------------------------|-------------------------------------------|
| Aligned Point Cloud      | `data/processed/test_aligned.ply`         |
| Deviation Heatmap        | `defects/deviation_heatmap.ply`           |
| Binary Defect Mask       | `defects/binary_defect_mask.ply`          |
| Metric Summary           | `reports/metrics.json`                    |
| Final Inspection Report  | `reports/inspection_report.json`          |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place STL Models

Put your reference and test `.stl` files in the `data/` directory:

### 3. Launch Pipeline

```bash
jupyter notebook notebook/inspection_pipeline.ipynb
```

Follow the notebook to:
1. Convert STL to PLY  
2. Clean and align clouds  
3. Compute deviations  
4. Save metrics and masks  

---

### 🧪 Synthetic Data Rendering

Render realistic training/inference data using Blender’s headless mode:

```bash
cd scripts
blender -b -P render.py
```

Outputs are saved in `blender_assets/outputs/`:
- RGB images  
- 16-bit depth maps  
- Normal maps  
- Camera intrinsics (JSON)  

---

## Requirements

Install via:

```bash
pip install -r requirements.txt
```
---

## Applications

- Mechanical QA automation  
- CAD-based geometry validation  
- Metrology in manufacturing  
- Research in 3D defect detection  
- Synthetic dataset generation for 3D vision models  

---

## What's Next?

- Add Mesh-to-Mesh Deviation (Not Just Point Clouds)
- Surface Defect Segmentation using Deep Learning
- Integration with Industrial 3D Scanners 

---

## 📩 Contact
I’m excited to connect and collaborate!  
- **Email**: [gbrohiith@gmail.com](mailto:your.email@example.com)  
- **LinkedIn**: [https://www.linkedin.com/in/rohiithgb/](https://linkedin.com/in/yourprofile)  
- **GitHub**: [https://github.com/GBR-RL/](https://github.com/yourusername)

---

## 📚 License
This project is open-source and available under the [MIT License](LICENSE).  

---

🌟 **If you like this project, please give it a star!** 🌟
