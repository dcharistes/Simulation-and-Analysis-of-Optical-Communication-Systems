# Simulation and Analysis of Optical Communication Systems

## Overview
This repository contains code and resources for simulating **optical communication systems**, focusing on signal modulation, and transmission through optical fiber. The simulation includes **Pulse Amplitude Modulation (PAM)**, using a Mach-Zehnder Modulator (MZM), simulating signal transmission through a linear fiber channel, and analyzing the impact of noise and dispersion. In this simulation, a digital signal is generated and modulated. Then we apply a None-Return-to-Zero (NRZ) pulse shaping and transmit it through an optical fiber. At the receiver, the signal is detected using a photodiode and processed to determine its quality. Key performance metrics, including **Bit Error Rate (BER)**, are computed and compared with theoretical values.

---

## Installation Instructions

### 1. Create a Virtual Environment (Recommended)
It is highly recommended to use a **virtual environment (venv)** to manage dependencies and avoid conflicts with system-wide Python packages.

#### For Windows (Bash):
```Bash
python -m venv .env
source .env\Scripts\activate
```

---

### 2. Install Dependencies
Once the virtual environment is activated, install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Usage
After setting up the environment and installing dependencies, you can run the simulation scripts.

```bash
python opticalCommTask1.py
```

---

## Repository Structure
- `lab1/` – Code for Lab 1 (Signal Modulation and BER Analysis)  
- `requirements.txt` – List of required Python packages  
- `README.md` – Documentation  

---
