<div align="center">
  <h1>Cognitive-Drive: A High-Fidelity Multi-modal Autonomous Driving Dataset & Value-Driven Data Curation Framework</h1>
  
  <a href="#">
    <img src="https://img.shields.io/badge/Dataset-LiDAR%20%7C%204D%20Radar%20%7C%20Camera-blue" alt="Dataset">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Scenarios-Adverse%20Weather%20%26%20Long--tail-red" alt="Scenarios">
  </a>
  <a href="https://github.com/open-mmlab/OpenPCDet">
    <img src="https://img.shields.io/badge/Codebase-OpenPCDet-green" alt="OpenPCDet">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-yellow" alt="License">
  </a>
</div>

<br/>

> **🚀 Note**: This repository currently hosts the **sample data and development kit** for the **Cognitive-Drive** dataset. This dataset is constructed via our proposed *Cognitive-Prism* active learning framework. The full detection benchmark and active learning codebase are under preparation and will be released upon publication.

## 📰 News
* **[2025-12-11]**: Initial release of the **Cognitive-Drive Development Kit**! 
* **[Coming Soon]**: The full **Cognitive-Drive** benchmark (7k+ annotated frames) will be available.
* **[Coming Soon]**: We will release the source code for the *Cognitive-Prism* active learning framework.

## ✨ Dataset Highlights

**Cognitive-Drive** is designed to stress-test autonomous perception systems in complex, open-world environments. Unlike static datasets, it is curated using a value-driven active learning approach.

- **🏎️ Multi-modal Sensor Suite**: High-resolution **Solid-state LiDAR**, **4D Millimeter-wave Radar**, and **8MP Camera**, fully synchronized and calibrated.
- **🌧️ Adverse Weather Focus**: Captures real-world signal degradation in heavy rain, night, and foggy conditions (over 12% rainy frames).
- **🎯 Safety-Critical Scenarios**: High density of Vulnerable Road Users (VRUs) and long-tail geometric anomalies mined from massive data streams.
- **🛠️ Value-Driven Curation**: All frames are selected by our *Cognitive-Prism* framework, ensuring high information density and geometric diversity.

## 1. Introduction

Robust 3D perception in complex urban scenarios remains a critical challenge for autonomous driving. While existing datasets (e.g., KITTI, nuScenes) have propelled the field forward, they often lack the **long-tail diversity** and **adverse weather coverage** required for deployment-grade reliability.

**Cognitive-Drive** bridges this gap. Collected using our custom platform in the dense urban environments of Shanghai, China, this dataset provides high-fidelity, synchronized multi-modal data. Key features of our collection include:
*   **Diverse Conditions**: Covering sunny, rainy, cloudy, and night-time driving.
*   **Rich Geometry**: Featuring non-repetitive scan patterns from solid-state LiDARs and distinct radar signatures.
*   **High-Quality Annotations**: Precise 3D bounding boxes for cars, pedestrians, and cyclists, verified manually.

This repository provides the **DevKit** (development kit) to help researchers visualize the data, parse annotations, and benchmark standard detectors (based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)).

<div align="center">
  <img src="https://github.com/user-attachments/assets/ff600486-1bd7-431d-9e81-a2b78887440c" width="60%" />
  <p><i>Figure 1. The sensor architecture of the Cognitive-Drive Dataset.</i></p>
</div>

## 2. Dataset & Acquisition Platform
### Sensor Configuration
Our autonomous driving research platform is equipped with a high-resolution Camera, a high-performance LiDAR, and a 4D Radar. All sensors are carefully calibrated and synchronized via GPS/PPS time. The detailed specifications are listed below:

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center">Sensor</th>
      <th colspan="3" style="text-align:center">Resolution</th>
      <th colspan="3" style="text-align:center">FOV</th>
    </tr>
    <tr>
      <th style="text-align:center">Range</th>
      <th style="text-align:center">Azimuth</th>
      <th style="text-align:center">Elevation</th>
      <th style="text-align:center">Range</th>
      <th style="text-align:center">Azimuth</th>
      <th style="text-align:center">Elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center"><strong>Camera</strong></td>
      <td style="text-align:center">/</td>
      <td style="text-align:center">3840px</td>
      <td style="text-align:center">2160px</td>
      <td style="text-align:center">/</td>
      <td style="text-align:center">120°</td>
      <td style="text-align:center">66°</td>
    </tr>
    <tr>
      <td style="text-align:center"><strong>Hybrid-solid State LiDAR</strong></td>
      <td style="text-align:center">2m</td>
      <td style="text-align:center">0.18°</td>
      <td style="text-align:center">0.24°</td>
      <td style="text-align:center">250m</td>
      <td style="text-align:center">120°</td>
      <td style="text-align:center">25°</td>
    </tr>
    <tr>
      <td style="text-align:center"><strong>4D Radar</strong></td>
      <td style="text-align:center">0.25m</td>
      <td style="text-align:center">2°</td>
      <td style="text-align:center">4°</td>
      <td style="text-align:center">150m</td>
      <td style="text-align:center">120°</td>
      <td style="text-align:center">20°</td>
    </tr>
  </tbody>
</table>

We extend the standard **KITTI-format** for our 3D annotations to include richer orientation details, while ensuring broad compatibility with existing tools.

*   **Coordinate System**: All 3D labels (`location`, `dimensions`) are provided in the **Camera Coordinate System**.
*   **Classes**: We focus on four main classes: `Car`, `Pedestrian`, `Cyclist`, and `Truck`.

<div align="center">
  <img src="https://github.com/user-attachments/assets/3c0f9a7b-a8cb-4e9a-a424-597e2c2c4013" width="45%" />
  <p><i>Figure 2. Object class distribution of the Cognitive-Drive dataset.</i></p>
</div>

### Label File Structure

Each line in a label file corresponds to a single object and contains 20 values, as detailed below.

```text
# Cognitive-Drive Label Format
# Values    Name         Description
---------------------------------------------------------------------------------------------------
   1        type         'Car', 'Pedestrian', 'Cyclist', 'Truck', etc.
   1        truncated    Float from 0.0 (non-truncated) to 1.0 (fully truncated).
   1        occluded     Integer (0,1,2,3) indicating occlusion state.
   1        alpha        Observation angle of object, ranging [-pi..pi].
   4        bbox         2D bounding box in the image: left, top, right, bottom.
   3        dimensions   3D object dimensions: height, width, length (h, w, l) in meters.
   3        location     3D object location x,y,z of the GEOMETRIC CENTER in CAMERA coordinates.
   3        rotation     Euler angles rx, ry, rz in the LIDAR coordinate system.
   1        score        Confidence score of the annotation.
   1        track_id     Tracking ID of the object.
   1        num_points   Number of LiDAR points within the 3D box.
```
## 3. Data Statistics

### Scenario Distribution

| Dimension | Detailed Breakdown |
| :--- | :--- |
| 🕒 **Time** | **Daytime** (81.4%), **Night** (13.3%), **Dawn** (5.3%) |
| 🌦️ **Weather** | **Sunny** (56.0%), **Cloudy** (24.8%), **Rainy** (15.2%), Foggy (4.0%) |
| 🏙️ **Area** | **Downtown** (41.7%), **Suburbs** (30.5%), Highway (12.9%), Tunnel (9.3%), Bridge (5.6%) |
| 🚦 **Traffic** | **Slow-Moving** (43.9%), **Free Flow** (37.7%), **Congested** (18.5%) |

### Stratified Spatial Distribution

<div align="center">
  <img src="https://github.com/user-attachments/assets/00fb45cd-5aa6-4c3e-b496-63009a2985c2" width="80%" />
  <p><i>Figure 3. Statistical distribution of annotated objects.</i></p>
</div>

## 4. Dataset Preparation

We are currently finalizing the data release process. The dataset structure is organized as follows. 
```text
Your_Project_Root
├── data
│   ├── tjsens                   # Your dataset root name
│   │   ├── gt_database          # Generated ground truth database for augmentation
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   │   ├── test.txt
│   │   ├── testing              # Testing split
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── radar
│   │   │   ├── velodyne
│   │   ├── training             # Training split
│   │   │   ├── calib            # Calibration files
│   │   │   ├── image_2          # Camera images
│   │   │   ├── label_2          # KITTI formatted labels
│   │   │   ├── radar            # 4D Radar point clouds
│   │   │   ├── velodyne         # LiDAR point clouds
```

## 5. Getting Started

### Environment Requirements
We have tested our framework on the following environment:
* **OS**: Ubuntu 20.04
* **Python**: 3.8+
* **PyTorch**: 1.13+ (CUDA 11.7 recommended)
* **OpenPCDet**: v0.6.0
* **spconv**: v2.x

### Installation

1.  **Clone this repository**
    ```bash
    git clone [https://github.com/RjieChen/Cognitive-Drive.git](https://github.com/RjieChen/Cognitive-Drive.git)
    cd your-repo-name
    ```

2.  **Create a conda environment**
    ```bash
    conda create -n tjsens python=3.8
    conda activate tjsens
    ```

3.  **Install PyTorch & Spconv (CUDA 11.7)**
    ```bash
    # Install PyTorch 1.13.0 with CUDA 11.7
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)

    # Install spconv-cu117
    pip install spconv-cu117
    ```

4.  **Install other dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Setup the project**
    ```bash
    python setup.py develop
    ```
    
## 6. Usage

### Data Info Generation

Before training, the dataset infos need to be generated using the following command:

```bash
# Generate data infos for training
python -m pcdet.datasets.tjsens.tjsens_dataset create_tjsens_infos tools/cfgs/dataset_configs/tjsens_dataset.yaml
```

### Training

  * **Single GPU Training**:

    ```bash
    python train.py --cfg_file cfgs/pv_rcnn.yaml
    ```

  * **Multi-GPU Distributed Training**:

    ```bash
    sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/pv_rcnn.yaml
    ```

### Evaluation

  * **Test with a pretrained checkpoint**:
    ```bash
    python test.py --cfg_file cfgs/pv_rcnn.yaml --batch_size 4 --ckpt path/to/checkpoint.pth
    ```

### Benchmark Results (3D AP)

We provide the baseline performance of state-of-the-art 3D object detectors on the **Cognitive-Drive** test set. All scores are reported as **3D Average Precision (AP)**.

<div align="center">

| Method | Car AP | Pedestrian AP | Cyclist AP | Truck AP | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: |
| PointRCNN | 36.00 | 17.37 | 32.78 | 21.08 | 26.81 |
| SECOND | 48.64 | 29.14 | 49.54 | 22.38 | 37.43 |
| PointPillars | 48.78 | 25.45 | 49.98 | 23.98 | 37.05 |
| CenterPoint | 51.47 | **37.06** | 56.09 | 29.39 | 43.50 |
| **PV-RCNN** | **55.46** | 36.05 | **57.04** | **33.55** | **45.53** |

</div>

> *Note: The mAP column represents the mean of the APs across the four classes. Best results are highlighted in bold.*

## 7. Visualization

<div align="center">
  <img src="https://github.com/user-attachments/assets/93174634-e3c7-446d-bf00-07a2d41d4ce4" alt="visualization" width="90%" />
  <p><i>Figure 4. Representing 3D annotations in multiple scenarios and sensor modalities. The three columns respectively display the projection of 3D annotation boxes in images, LiDAR point clouds and 4D radar detection.</i></p>
</div>


## 8. Acknowledgement

Our code is built upon [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). We thank the authors for their open-source contribution.


## 9. Contact

For any questions, please feel free to contact:

  * **[Ruijie CHEN]**: [ruijiechen@tongji.edu.cn]


