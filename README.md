# 🪟 Edge-to-Cloud Smart Mosquito Trap for Automated Vector Monitoring

This repository contains the full pipeline for an AI-powered mosquito trap system that integrates **edge computing**, **cloud-based image processing**, and **automated public health surveillance**. Built around a **Raspberry Pi-based trap** and a **Linode-hosted server**, this project enables real-time detection, classification, and alerting of disease-vector mosquitoes.

---

## 🫠 Project Description

The smart mosquito trap uses a **fan and light mechanism** to attract mosquitoes into a chamber where they stick to a pad. A **Raspberry Pi camera** captures **10 multi-focus images** of the sticky pad and uploads them to **Google Firestore (NoSQL)**.

A server-side Python script continuously monitors Firestore for new uploads, and when new data is detected, it triggers a multi-step image processing pipeline to isolate, enhance, and classify individual mosquitoes. If a disease-vector species is identified, it sends **automated alerts** to designated **public health officials**.

---

## 🔄 End-to-End Pipeline

![Pipeline Diagram](https://github.com/FarhatBuet14/Smart-AI-Mosquito-Trap/raw/main/Pipeline.png)


### Edge Device (Raspberry Pi)

- Triggers fan + light to lure mosquitoes
- Captures 10 multi-focus images
- Uploads image stack to Google Firestore with metadata

### Cloud Server (Linode)

1. **Undistortion**: Applies fisheye correction to each focus image
2. **Alignment**: Aligns all focus images to the reference (`focus_0`) using ECC
3. **Mosquito Detection**: Applies a trained **Faster-RCNN** model to localize individual mosquitoes
4. **Fusion**: Uses **Laplacian pyramid fusion** or **Gaussian-weighted stack** to combine multi-focus crops into a single sharp image
5. **Deblurring**: Applies either classical (Wiener filter) or deep learning-based methods (DeblurGAN-v2 or MIMO-UNet)
6. **Sharpening**: Enhances anatomical textures via unsharp masking
7. **Classification**: Predicts species label using a fine-tuned **NasNetMobile** model
8. **Alerting**: If a disease-vector mosquito (e.g., *An. stephensi*) is detected, sends email alerts to public health contacts
9. **Storage**: Saves predictions and metadata in **PostgreSQL** for query and dashboard visualization

---

## 📁 Repository Structure

```
mosquito-ai-trap/
│
├── edge_device/
│   └── capture_and_upload.py         # Raspberry Pi image capture & upload
│
├── server_pipeline/
│   ├── watcher.py                    # Firestore listener and processor
│   ├── process_pipeline.py          # Main orchestration of all steps
│   ├── models/
│   │   ├── faster_rcnn.pth          # Pretrained mosquito detector
│   │   └── nasnet_mobile.pth        # Pretrained species classifier
│
├── utils/
│   ├── fusion.py                    # Laplacian and weighted fusion
│   ├── deblur.py                    # Classical & deep learning deblurring
│   ├── align.py                     # ECC-based image alignment
│   └── undistort.py                 # Fisheye distortion correction
│
├── database/
│   └── postgres_logger.py          # PostgreSQL integration
│
└── alert/
    └── notifier.py                 # Email alert module
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/FarhatBuet14/Smart-AI-Mosquito-Trap.git
cd mosquito-ai-trap
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

- Add your Firebase credentials to `firebase_config.json`
- Add PostgreSQL credentials to `.env`

### 4. Run Edge Script on Raspberry Pi

```bash
python edge_device/capture_and_upload.py
```

### 5. Run Cloud Pipeline on Server

```bash
python server_pipeline/watcher.py
```

---

## 📂 Pretrained Models

| Model        | Description                                          | Download                                                           |
| ------------ | ---------------------------------------------------- | ------------------------------------------------------------------ |
| Faster-RCNN  | Mosquito detection model trained on custom dataset   | [Download](https://your-link.com/faster_rcnn.pth)                  |
| EfficientNET | Species classification model (binary or multi-class) | [Download](https://your-link.com/nasnet_mobile.pth)                |
| DeblurGANv2  | Optional deep deblurring model                       | [Link to official repo](https://github.com/VITA-Group/DeblurGANv2) |

---

## 🔗 Dataset Reference

- **Detection Dataset**: 5,000+ labeled mosquito images with bounding boxes
- **Classification Dataset**: Cropped mosquito images across 8 species
- **Focus Stacks**: Each sample includes 10 images at varying focus levels

*(For access to datasets or collaboration inquiries, contact the maintainer.)*

---

## 📈 Database Design

- **Firestore (NoSQL)**: Stores raw uploads and metadata from edge devices
- **PostgreSQL**: Stores final mosquito species, confidence scores, image links, and timestamps for efficient querying and reporting

---

## 🚨 Alert System

- Uses SMTP/SendGrid to send alert emails when target vector species (e.g., *An. stephensi*) is detected
- Emails include image thumbnails, confidence scores, and location info

---

## 📅 Use Cases

- Real-time public health surveillance
- Vector monitoring in urban & rural settings
- AI model benchmarking with real-world environmental data

---

## 🙋 Contributing

Contributions are welcome! If you want to improve the pipeline, add new models, or integrate real-time dashboards, feel free to open a pull request.

---

## 📅 License

MIT License — free to use, modify, and distribute.

---


## 🧳 Acknowledgments

This project was supported by research on AI-based mosquito classification for vector surveillance and disease prevention. Special thanks to the SCoRE Lab at University of South Florida and the global entomology and computer vision communities.

