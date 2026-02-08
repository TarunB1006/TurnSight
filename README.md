<div align="center">

# 🚦 TurnSight

### AI-Powered Vehicle Detection, Tracking & Traffic Forecasting System

[![Competition Winner](https://img.shields.io/badge/🏆_Phase_1_Winner-Bengaluru_Mobility_Challenge_2024-gold?style=for-the-badge)](https://ieee-dataport.org/competitions/bengaluru-mobility-challenge-2024)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg?style=flat-square&logo=yolo)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU_Support-76B900.svg?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg?style=flat-square)](./LICENSE)

**Team GetFined** | RV College of Engineering

</div>

---

## 👥 Team Members

| Name | Role |
|------|------|
| **Tarun Bhupathi** | Team Lead & Developer |
| **Sundarakrishnan N** | ML Engineer |
| **Sohan Varier** | Computer Vision Engineer |
| **Manaswini Simhadri Kavali** | Data Scientist |

---

## 📚 Table of Contents

- [About the Project](#-about-the-project)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Program Scripts](#-program-scripts)
- [Development Scripts](#-development-scripts)
- [Docker Deployment](#-docker-deployment)
- [System Requirements](#-system-requirements)
- [Open-Source Tools](#-open-source-tools)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [Resources](#-resources)

---

## 🎯 About the Project

TurnSight is an advanced AI-powered system designed to analyze traffic patterns from camera feeds, providing real-time vehicle detection, tracking, and short-term traffic forecasting. This project won **Phase 1** of the prestigious **Bengaluru Mobility Challenge 2024**.

### 🎯 Problem Statement

Participants were provided with camera feeds from 23 Safe City cameras in northern Bengaluru (around the IISc campus). The challenge was to:
- Provide short-term predictions (30 minutes into the future) of vehicle counts by vehicle type
- Predict vehicle turning patterns at junctions
- Make predictions at different points from where camera feeds are available

### 📖 Documentation

- **📄 Detailed Report**: [View Report](https://drive.google.com/file/d/1YZztqHRN1J5TLh3QNKnYMgrsRQ3Dhf7d/view?usp=drive_link)
- **🌐 Event Details**: [Bengaluru Mobility Challenge 2024](https://dataforpublicgood.org.in/bengaluru-mobility-challenge-2024/)

---

## 🛠 Tech Stack

<div align="center">

| Technology | Purpose |
|------------|---------|
| ![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black) | Real-time object detection |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core programming language |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) | Video processing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | Data manipulation |
| ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) | Containerization |
| ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) | GPU acceleration |
| ![ARIMA](https://img.shields.io/badge/ARIMA-FF6F00?style=for-the-badge) | Time series forecasting |

</div>

---

## 🔄 Architecture

Our pipeline follows a streamlined approach from input to prediction:

```mermaid
graph LR
    A[📹 Video Input] --> B[🎯 YOLOv8 Detection]
    B --> C[🔍 Object Tracking]
    C --> D[📊 Vehicle Counting]
    D --> E[📈 ARIMA Forecasting]
    E --> F[📤 JSON Output]
    
    style A fill:#e1f5ff
    style B fill:#b3e5fc
    style C fill:#81d4fa
    style D fill:#4fc3f7
    style E fill:#29b6f6
    style F fill:#03a9f4
```

**Pipeline Stages:**
1. **Video Input** - Process camera feeds from multiple locations
2. **YOLOv8 Detection** - Detect 7 classes of vehicles in real-time
3. **Object Tracking** - Track vehicles across frames
4. **Vehicle Counting** - Count vehicles and turning patterns at junctions
5. **ARIMA Forecasting** - Predict future vehicle counts (30 min ahead)
6. **JSON Output** - Generate structured output for analysis

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (recommended for faster processing)
- CUDA Toolkit (for GPU support)
- 10 GB free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TarunB1006/TurnSight.git
   cd TurnSight
   ```

2. **Install dependencies**
   ```bash
   cd "Program Scripts( Submission)"
   pip3 install -r requirements.txt
   ```

### Usage

1. **Prepare your input JSON file**

   Create an `input.json` file with the following format:
   ```json
   {
      "Cam_ID": {
         "Vid_1": "/app/data/Cam_ID_vid_1.mp4",
         "Vid_2": "/app/data/Cam_ID_1_vid_2.mp4"
      }
   }
   ```

2. **Run the pipeline**
   ```bash
   python3 app.py input.json output.json
   ```

3. **View results**
   
   The predictions will be saved in `output.json` with vehicle counts and turning patterns.

---

## 📦 Program Scripts

The main pipeline consists of these essential files:

<details>
<summary><b>Click to expand script details</b></summary>

| Script | Description | Type |
|--------|-------------|------|
| `app.py` | Main driver code - entry point for the pipeline | 🎯 Core |
| `best.pt` | Trained YOLOv8 model for 7 vehicle classes | 🤖 Model |
| `config.py` | Junction coordinates for turning pattern detection | ⚙️ Config |
| `outputTemplate.py` | Output format dictionary (JSON structure) | 📋 Template |
| `customCounter.py` | Modified Ultralytics counter for custom requirements | 🔧 Utility |
| `video_processor.py` | VideoProcessor class for detection, tracking & counting | 🎬 Core |
| `forecasting.py` | Forecaster class with ARIMA prediction logic | 📈 Core |
| `output_handler.py` | Process outputs into required dictionary format | 📤 Utility |

</details>

> **Note:** Only `app.py` should be executed directly. Other files are imported as modules.

---

## 🔬 Development Scripts

Additional scripts used during development (not required for pipeline execution):

<details>
<summary><b>Click to expand development tools</b></summary>

| Script | Purpose |
|--------|---------|
| `extract_images.py` | Extract frames from videos for training dataset |
| `auto_annotate.py` | Automated annotation using base model |
| `data_split.py` | Split dataset into train/test/validation sets |
| `stream.py` | Real-time visualization of YOLO predictions |
| `capture_coordinates.py` | Interactive tool to capture junction box coordinates |
| `view.py` | Visualize turning boxes on junction images |
| `data.yaml` | Dataset configuration for model training |
| `predict_arima.py` | Test various ARIMA forecasting parameters |
| `data_combine.py` | Merge annotated datasets from team members |

</details>

---

## 📋 Requirements

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **opencv-python-headless** | 4.10.0.84 | Video file processing |
| **ultralytics** | 8.2.58 | YOLOv8 model framework |
| **pandas** | 1.5.3 | Data structure and manipulation |
| **pmdarima** | 2.0.4 | Auto-ARIMA forecasting |
| **Shapely** | 2.0.6 | Geometric operations (Ultralytics dependency) |
| **statsmodels** | 0.14.2 | Statistical forecasting methods |

**Installation:**
```bash
pip3 install -r requirements.txt
```

### Additional Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | 3.7.1 | Data visualization |
| **numpy** | 2.1.0 | Array operations |
| **prophet** | 1.1.5 | Facebook's forecasting model |
| **scikit-learn** | 1.0.2 | ML utilities and data splitting |
| **opencv-python** | 4.10.0.84 | Video display (non-headless) |

---

## 🐳 Docker Deployment

TurnSight is fully containerized with GPU support for production deployment.

### Building the Docker Image

```bash
docker build -t username/turnsight:latest .
```

### Pushing to Registry

```bash
docker push username/turnsight:latest
```

### Running the Container

```bash
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -v /your/data/path:/app/data \
  username/turnsight:latest \
  python3 app.py input.json output.json
```

**Parameters:**
- `--runtime=nvidia --gpus all`: Enable GPU acceleration
- `-v /your/data/path:/app/data`: Mount your data directory
- `input.json output.json`: Input and output file paths

---

## 💻 System Requirements

| Component | Minimum Specification |
|-----------|----------------------|
| **CPU** | Intel Core i5 or equivalent |
| **GPU** | NVIDIA GTX 1650 (4GB VRAM) |
| **RAM** | 8 GB |
| **Storage** | 10 GB free space |
| **OS** | Linux / Windows / macOS |

> **GPU Memory Usage:** ~1GB VRAM required for real-time inference

---

## 🔓 Open-Source Tools

This project is built on top of excellent open-source technologies:

- **[YOLOv8](https://github.com/ultralytics/ultralytics)** by Ultralytics - Real-time object detection and image segmentation
- **[LabelImg](https://github.com/HumanSignal/labelImg)** - Annotation tool for creating YOLO-format bounding boxes

---

## 📖 Citation

If you use this software in your research or project, please cite it as:

```bibtex
@software{Phase1_BMC_GetFined,
  author = {Sundarakrishnan N and Sohan Varier and Tarun Bhupathi and Manaswini Simhadri Kavali},
  title = {Phase1-BMC-GetFined},
  version = {1.0.0},
  date = {2024-09-23},
  url = {https://github.com/TarunB1006/TurnSight}
}
```

> **Note:** This repository was originally published as Phase1-BMC and is now maintained as TurnSight.

---

## 🙏 Acknowledgments

We would like to express our gratitude to:

- **IEEE DataPort** and **Data for Public Good** for organizing the Bengaluru Mobility Challenge 2024
- **Ultralytics** for the excellent YOLOv8 framework
- **RV College of Engineering** for their support and resources
- The open-source community for the amazing tools that made this project possible

---

## 📚 Resources

- **Competition Page**: [Bengaluru Mobility Challenge 2024](https://ieee-dataport.org/competitions/bengaluru-mobility-challenge-2024)
- **Event Website**: [Data for Public Good](https://dataforpublicgood.org.in/bengaluru-mobility-challenge-2024/)
- **Detailed Report**: [Google Drive](https://drive.google.com/file/d/1YZztqHRN1J5TLh3QNKnYMgrsRQ3Dhf7d/view?usp=drive_link)
- **Repository**: [GitHub](https://github.com/TarunB1006/TurnSight)

---

<div align="center">

**Made with ❤️ by Team GetFined**

🏆 Phase 1 Winners | Bengaluru Mobility Challenge 2024

[![GitHub](https://img.shields.io/badge/GitHub-TurnSight-black?style=flat-square&logo=github)](https://github.com/TarunB1006/TurnSight)
[![Competition](https://img.shields.io/badge/Competition-BMC_2024-blue?style=flat-square)](https://ieee-dataport.org/competitions/bengaluru-mobility-challenge-2024)

</div>
