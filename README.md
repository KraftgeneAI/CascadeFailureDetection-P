# Pipeline Digital Twin: Cascade Failure Prediction

A physics-based digital twin and machine learning pipeline designed to simulate, detect, and predict catastrophic cascade failures in fluid networks (e.g., oil, gas, and water pipelines). 

This project fuses hydraulic fluid dynamics (steady-state flow, transient surge, and thermal dynamics) with deep learning (Graph Neural Networks and LSTMs) to predict failure probabilities and time-to-failure in real-time.

## Current Features

* **Physics-Based Simulator:** Solves the hydraulic Laplacian matrix to simulate fluid flow, pressure drops, and friction heat across a 118-node topology.
* **Cascade Engine:** Simulates real-world pipeline failures including overpressure ruptures, low-pressure cavitation, and flow capacity limits.
* **Synthetic Data Generator:** Batch orchestrator that generates perfectly labeled `.pkl` datasets of "Normal", "Stressed", and "Cascade" scenarios for ML training.

## Installation

Ensure you have Python 3.9+ installed. Clone the repository and install the dependencies:

```bash
git clone [https://github.com/YOUR-USERNAME/CascadeFailureDetection-P.git](https://github.com/YOUR-USERNAME/CascadeFailureDetection-P.git)
cd CascadeFailureDetection-P
pip install -r requirements.txt