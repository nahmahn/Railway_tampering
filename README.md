# Railway Tampering & Anomaly Detection System

![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview
Our solution is a **Meta-Model–based Mixture-of-Experts architecture** designed for real-time detection of anomalies and intentional track tampering across large rail networks. By fusing structural, visual, thermal, and contextual evidence, the system performs geo-referenced anomaly analysis, pinpoints affected track segments and hotspots, and generates actionable outputs including real-time alerts, visualizations, and operational action reports.

---

## 🛠 Tech Stack

### Frontend
![Next.js](https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Tailwind CSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Recharts](https://img.shields.io/badge/Recharts-22b5bf?style=for-the-badge&logo=react&logoColor=white)

### Backend & AI
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

### Data & Tools
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-77ACF1?style=for-the-badge&logo=seaborn&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

---

## Key Features

### Real-Time Anomaly & Tampering Detection
Continuous monitoring enables early detection of abnormal vibrations, structural deviations, thermal hotspots, foreign object presence, and unauthorized human activity before these escalate into safety-critical incidents.

### Real-Time Alerts with Geo-Referenced Visualization
All detections are geo-tagged and visualized on a map-based interface, highlighting impacted zones and hotspots. Alerts and notifications are pushed in real time to support rapid, localized response.

### Automated Formal Incident Reporting
The system generates **Professional Government-Standard PDF Reports** for every incident. These reports include:
*   **Executive Summary & Technical Assessment**: AI-generated detailed analysis of the fault using expert terminology.
*   **Visual Evidence & Analysis**: High-resolution charts (Time Series, Heatmaps) generated via Matplotlib/Seaborn to visualize vibration anomalies.
*   **Remedial Action Plan**: Structured tables detailing urgency, ownership, and specific engineering steps.
*   **Official Formatting**: Includes Government of India and Ministry of Railways branding, making reports ready for immediate official circulation.

### Multi-Sensor, Multi-Stream Intelligence
The system seamlessly handles diverse input streams: geometric sensors, accelerometers, DAS, CCTV, drones, thermal cameras, and LiDAR without requiring uniform coverage. Each modality contributes complementary evidence, enabling reliable detection even when some sensors are unavailable.

### Comprehensive Tampering Analysis
The system detects intentional track tampering by correlating structural, vibration, visual, and thermal cues. It identifies scenarios such as:
*   Component manipulation
*   Foreign object placement
*   Rail cutting
*   Ballast disturbance
*   Signaling or cable tampering

---

## Unique Selling Points (USP)

### 1. Evidence-Driven Mixture-of-Experts Architecture
Unlike monolithic models, the system dynamically selects expert reasoning paths based on available evidence, enabling robustness, scalability, and explainability across large rail networks.

### 2. Operationally Actionable Outputs
The system goes beyond detection by generating clear action reports, mitigation steps, and **rerouting suggestions** that align with real railway operations.

### 3. Deep Geo-Spatial Intelligence
All anomalies are geo-referenced, enabling precise localization, hotspot identification, and impact assessment across vast and distributed infrastructure.

### 4. Human-in-the-Loop Friendly Design
Natural language queries, visual explanations, and contextual reasoning make the system usable by operators without requiring ML expertise.

### 5. Designed for Sparse, Real-World Deployments
The architecture assumes imperfect sensor coverage and varying data quality, making it practical for nationwide railway adoption rather than controlled environments.

---

## Getting Started

### Prerequisites
*   Node.js 18+
*   Python 3.9+
*   Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/nahmahn/Railway_tampering.git
    cd Railway_tampering
    ```

2.  **Frontend Setup**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

3.  **Backend Setup**
    ```bash
    # (Optional) Create a virtual environment
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate

    pip install -r backend/requirements.txt
    ```

## Demonstration
Check out our live demonstration here: [Project Demo](https://shorturl.at/MmCfg)

---
*Built for Hack4Delhi*
