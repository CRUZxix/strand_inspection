# Strand Inspection System
### AI-Assisted Automated Strand Detection and Quality Inspection

An image-based inspection system designed to automatically detect components inside a packet, identify their strand counts, and determine whether each component meets the required quality criteria.

The system aims to reduce manual inspection effort by combining computer vision, AI-assisted image analysis, and automated inspection reporting.

---

## Project Overview

The Strand Inspection System takes an image of a packet containing multiple strand-based components and performs automated inspection.

The system is designed to:

- Identify individual components within a packet.
- Automatically generate bounding boxes around detected components.
- Estimate the number of strands in each component.
- Assign component positions using row and column identification.
- Classify components as GOOD, DEFECT, or UNCERTAIN.
- Generate a visual inspection result and a final component-wise inspection matrix.

## Key Features

| Feature | Description |
|---|---|
| Image Upload | Upload packet images for inspection |
| Automatic Detection | Identify individual components in the packet |
| Bounding Boxes | Visualize detected components on the original image |
| Strand Counting | Estimate the number of strands in each component |
| Position Identification | Assign row and column identifiers such as R1C1 |
| Defect Classification | Identify components that do not meet inspection criteria |
| Inspection Matrix | Display component-wise results in a colour-coded matrix |
| Visual Reporting | Present inspection results in an easy-to-understand format |

## Inspection Workflow

The intended inspection workflow consists of the following stages:

1. **Image Acquisition:** Upload an image of the packet.
2. **Component Detection:** Locate individual components and generate bounding boxes.
3. **Component Identification:** Assign row and column positions to detected components.
4. **Strand Counting:** Estimate the number of strands in each component.
5. **Quality Evaluation:** Compare the predicted count against the configured acceptance criteria.
6. **Final Inspection:** Generate a colour-coded inspection matrix.

## Final Inspection Output

The system presents the results in a component-wise matrix.

Example:

| Component | Strand Count | Status |
|---|---:|---|
| R1C1 | 2 | 🔴 DEFECT |
| R1C2 | 3 | 🟢 GOOD |
| R1C3 | 3 | 🟢 GOOD |
| R2C1 | 3 | 🟢 GOOD |
| R2C2 | 3 | 🟢 GOOD |
| R2C3 | 3 | 🟢 GOOD |

*Illustrative results only. Actual values are generated from the inspection process.*

The overall packet is classified as DEFECT DETECTED if one or more components fail the configured inspection criteria.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Interactive web application |
| OpenAI API | AI-assisted image analysis |
| Computer Vision | Image processing and component localization |
| YOLO | Object-detection model experimentation |
| Pillow / OpenCV | Image processing |
| Pandas | Dataset and inspection data handling |

## Dataset Structure

The project uses a packet inspection dataset containing:

- Packet images.
- Component-level inspection labels.
- Strand-count information.
- Row and column positions.
- GOOD / DEFECT status.
- Defect-type information.

The current dataset contains 67 packet images and 670 component-level records.

### Label Information

The component-level annotation CSV contains the following fields:

| Field | Description |
|---|---|
| image_id | Unique image identifier |
| image_filename | Original image filename |
| captured_at | Image capture timestamp |
| row | Component row position |
| column | Component column position |
| position | Component position identifier |
| strand_count | Number of strands |
| status | GOOD / DEFECT |
| defect_type | Type of defect, where applicable |

## Project Structure

```text
strand-inspection/
│
├── .streamlit/
│   └── secrets.toml
│
├── images/
│   └── Packet inspection images
│
├── runs/
│   └── Model experiment outputs
│
├── app.py
├── verify_dataset.py
├── labels.csv
├── packets.csv
├── yolo26n.pt
├── requirements.txt
├── .gitignore
└── README.md
```

*The project structure may evolve as development continues.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/CRUZxix/strand_inspection.git
```

### 2. Navigate to the Project

```bash
cd strand_inspection
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not yet available, install the dependencies used by the application individually.

### 5. Configure API Credentials

Create a file named `secrets.toml` inside the `.streamlit` directory.

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Never commit API keys or private credentials to GitHub.

### 6. Run the Application

```bash
python -m streamlit run app.py
```

Open the local application:

```text
http://localhost:8501
```

## Development Goals

The project is being developed toward a more reliable automated inspection pipeline.

Planned areas of improvement include:

- Improving component localization and bounding-box accuracy.
- Developing a dedicated object-detection model for the components.
- Improving strand-count estimation.
- Handling reflections, transparent packaging, and image orientation.
- Improving inspection confidence and uncertainty handling.
- Evaluating performance using unseen packet images.
- Improving the final inspection interface and reporting.

## Repository

GitHub: https://github.com/CRUZxix/strand_inspection

## Author


*This project is under active development. Detection and classification performance are subject to ongoing testing and validation.*
