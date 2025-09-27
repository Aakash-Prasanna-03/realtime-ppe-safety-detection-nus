
# Face Detection and Violation Tracking System

![Sample Detection Output](outputs/screenshots/sample_detection1.png)

## Overview
This project is an advanced automated face detection and violation tracking system designed for real-time monitoring and reporting of unauthorized or unknown individuals in sensitive environments. It leverages state-of-the-art deep learning models (MTCNN, InceptionResnetV1, FastRCNN, YOLO) to detect faces and objects in images, compare them with registered users, and log security violations.

## Why This Project is Important for Security
- **Access Control:** Prevents unauthorized access to restricted areas by identifying unknown individuals.
- **Incident Documentation:** Automatically logs and stores evidence of security violations, supporting investigations.
- **Real-Time Alerts:** Enables rapid response to security breaches or policy violations.
- **Scalability:** Can be deployed across multiple cameras and locations, supporting large-scale security operations.
- **Automation:** Reduces manual monitoring workload, increasing efficiency and accuracy.

## Example Use Cases
- Office buildings and corporate campuses
- Construction sites (PPE and safety compliance)
- Data centers and server rooms
- Schools and universities
- Public infrastructure and transport hubs

## How It Works
1. **Face Detection:** Detects faces in input images or video frames using MTCNN.
2. **Face Recognition:** Extracts embeddings with InceptionResnetV1 and compares them to known users.
3. **Unknown Tracking:** Assigns unique labels to unknown individuals and tracks their appearances.
4. **Violation Logging:** Records each violation in a CSV report, including image evidence.
5. **Object Detection:** (Optional) Uses YOLO/FastRCNN for detecting safety equipment, objects, or other compliance factors.

## Example Outputs

| Multi-Object Detection (YOLO/FastRCNN) | Safety Cone Detection (YOLO) |
|:--------------------------------------:|:---------------------------:|
| ![Multi-Object](outputs/screenshots/sample_detection1.png) | ![Safety Cones](outputs/screenshots/sample_detection2.png) |

*Above: Example outputs showing object and safety cone detection with bounding boxes.*


## Project Structure
```
│
├── data/
│   ├── images/
│   ├── labels/
│   └── user_profiles/
├── models/
│   ├── yolo/
│   ├── fastrcnn/
│   └── embeddings/
├── scripts/
│   ├── train_yolo.py
│   ├── train_fastrcnn.py
│   ├── inference_yolo.py
│   ├── inference_fastrcnn.py
│   ├── model_comparison.py
│   └── cuda_check.py
├── notebooks/
│   ├── Fastrcnn_model.ipynb
│   └── ...
├── outputs/
│   ├── screenshots/
│   ├── reports/
│   └── ...
```

## Installation

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Train FastRCNN

1. Ensure your training and validation data are in the correct folders:
   - `data/train/images/`, `data/train/labels/`
   - `data/valid/images/`, `data/valid/labels/`
2. Run the training script:
   ```
   python scripts/train_fastrcnn.py
   ```
3. The trained model will be saved to `models/fastrcnn/final_model2.pth`.
4. Training and validation loss graphs will be saved to `outputs/`.


## Results and Metrics (Coming Soon)

**Detailed results, per-class metrics, and evaluation tables will be added here after the next commit.**

When you run the training script, it will automatically generate and save detailed graphs of training and validation loss in the `outputs/` directory. You can use these graphs to monitor model performance and convergence.

## How to Run Inference

Use the provided inference scripts in `scripts/` to test the trained model on new images or video streams. See script docstrings for usage.
├── README.md
├── requirements.txt
└── .gitignore
```


**Note:** All scripts, models, data, and outputs are now organized for clarity and scalability. Results and evaluation tables will be updated soon.


## Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <project-root>
```

### 2. Set up your environment
Install dependencies:
```bash
pip install -r requirements.txt
# or, for conda:
conda env create -f environment.yml
conda activate <env-name>
```

### 3. Prepare your data
- Place all images in `data/images/`
- Place all label files in `data/labels/`
- Place user profile images in `data/user_profiles/`
- Place screenshots for detection in `outputs/screenshots/`

### 4. Add model weights and embeddings
- Place YOLO weights in `models/yolo/`
- Place FastRCNN weights in `models/fastrcnn/`
- Place face embeddings in `models/embeddings/`

### 5. Run scripts and notebooks
- Use scripts in `scripts/` for training, inference, and evaluation:
   - `python scripts/facedetection.py` — Run face detection and violation logging
   - `python scripts/modeltest.py` — Test FastRCNN model
   - `python scripts/test2.py` — Additional test script
   - `python scripts/model_comparison.py` — Compare YOLO and FastRCNN models
   - `python scripts/cuda_check.py` — Check CUDA/GPU availability
- Use notebooks in `notebooks/` for interactive exploration and training

### 6. View outputs
- Reports and CSVs: `outputs/reports/`
- Screenshots: `outputs/screenshots/`
- Confusion matrices and plots: `outputs/confusion_matrices/`

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


---

*This project demonstrates the power of AI-driven security automation, providing a scalable, efficient, and reliable solution for modern safety and compliance needs.*


## Notebooks
- `notebooks/main_file.ipynb` — Main workflow
- `notebooks/Fastrcnn_model.ipynb` — FastRCNN model notebook
- `notebooks/face_detection.ipynb` — Face detection notebook

## Key Scripts
- `scripts/facedetection.py` — Main face detection and violation logging
- `scripts/uidesign.py` — UI design
- `scripts/modeltest.py` — Model testing
- `scripts/test2.py` — Additional test script
- `scripts/model_comparison.py` — Model comparison
- `scripts/cuda_check.py` — CUDA check script


## Outputs
- `outputs/reports/violations_report.csv` — CSV report of violations
- `outputs/screenshots/` — Processed screenshots (auto-cleared)
- `outputs/` — All graphs, confusion matrices, and evaluation tables (to be updated)

## Extending the System
- Integrate with real-time video streams for live monitoring
- Add email/SMS alerting for critical violations
- Connect to access control systems for automated lockdowns
- Expand object detection to include PPE, vehicles, or other compliance items

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


---

*This project demonstrates the power of AI-driven security automation, providing a scalable, efficient, and reliable solution for modern safety and compliance needs.*


"# realtime-ppe-safety-detection-nus" 
