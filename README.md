# Brain Tumor Segmentation and 3D Visualization

A research-oriented web application for automated brain tumor segmentation from MRI NIfTI volumes. The app provides 2D overlay visualization, multimodal MRI inspection, interactive 3D visualization, ET / ED / NET region volume analysis, and separate tumor foci metadata.

## Features

- Brain tumor segmentation from NIfTI MRI volumes
- ET / ED / NET region separation
- Total tumor volume estimation
- Separate connected tumor foci detection
- Sub-threshold segmentation island filtering
- Per-focus metadata and isolated mask previews
- Axial, coronal, and sagittal overlay viewer
- Multimodal axial viewer for FLAIR, T1, T1ce, and T2
- Interactive 3D visualization
- Metadata JSON export
- Research portfolio-style frontend UI

## Research Use Only

This project is intended for educational and research visualization only. It is not intended for clinical diagnosis, treatment planning, or replacement of expert radiological review.

## Project Structure

```text
backend/
  app.py
  requirements.txt
  model/
    infer_wrapper.py
    PLACE_MODEL_HERE.txt

frontend/
  index.html
  styles.css
  app.js

screenshots/
  hero.png
  overlay-viewer.png
  tumor-foci.png
  3d-viewer.png

## Model Checkpoint

The trained model checkpoint is not included in this repository because model files are large.
Place your trained checkpoint at:
   backend/model/unet3d_best.pt

## Setup
   From the project root:
      cd backend
      python -m venv .venv311
      .\.venv311\Scripts\activate
      pip install -r requirements.txt
## Run

From the backend folder:
   uvicorn app:app --host 0.0.0.0 --port 7860