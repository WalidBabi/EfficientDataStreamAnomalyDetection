# Efficient Data Stream Anomaly Detection

## Project Overview
Your task is to develop a Python script capable of detecting anomalies in a continuous data stream. This stream, simulating real-time sequences of floating-point numbers, could represent various metrics such as financial transactions or system metrics. Your focus will be on identifying unusual patterns, such as exceptionally high values or deviations from the norm.

## Features
- **Algorithm Selection**: Uses Gaussian Distribution for anomaly detection, suitable for adapting to concept drift and seasonal variations.
- **Data Stream Simulation**: Emulates a data stream with regular patterns, seasonal elements, and random noise.
- **Real-Time Anomaly Detection**: Implements a real-time mechanism to accurately flag anomalies in the data stream.
- **Optimization**: Optimized for both speed and efficiency.
- **Visualization**: Provides a real-time visualization tool to display the data stream and detected anomalies.

## Requirements
- Python 3.12.7
- `numpy`
- `matplotlib`

Install the required libraries using:
```bash
pip install -r requirements.txt
