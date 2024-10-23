import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def simulate_data_stream():
    """
    Simulates a continuous data stream of floating-point numbers with seasonal patterns, noise, and regular patterns.

    Yields:
    float: Next data point in the data stream.
    """
    np.random.seed(42)
    time_index = 0
    while True:
        seasonal_pattern = 10 * np.sin(2 * np.pi * time_index / 100)
        noise = np.random.normal(0, 10)
        trend = 0.1 * time_index
        data_point = 50 + seasonal_pattern + noise + trend
        yield data_point
        time_index += 1

def detect_anomalies_streaming(data_stream, window_size=100, threshold=0.005, batch_size=10):
    """
    Detects anomalies in a streaming data sequence using Gaussian Distribution.

    Args:
    data_stream (iterable): Continuous data stream.
    window_size (int): Number of data points in the sliding window.
    threshold (float): Threshold for anomaly detection based on PDF value.
    batch_size (int): Number of points processed in each batch.

    Returns:
    list: List of tuples containing the index and value of detected anomalies.
    """
    window = []
    anomalies = []
    buffer = []

    for i, point in enumerate(data_stream):
        buffer.append(point)
        if len(buffer) == batch_size:
            window.extend(buffer)
            buffer = []

            if len(window) > window_size:
                window = window[-window_size:]  # Maintain the window size

            if len(window) == window_size:
                mean = np.mean(window)
                std_dev = np.std(window)

                if std_dev > 0:
                    for j, p in enumerate(window[-batch_size:]):
                        pdf_value = (1 / (np.sqrt(2 * np.pi * (std_dev**2)))) * np.exp(-(p - mean)**2 / (2 * (std_dev)**2))
                        if pdf_value < threshold:
                            anomalies.append((i - batch_size + j, p))
                            print(f"Anomaly detected at point {i - batch_size + j}: {p}")

    return anomalies

def plot_anomalies(data, anomalies, ax):
    """
    Updates the plot with the latest data point and highlights detected anomalies.

    Args:
    data (list): List of all data points up to the current time.
    anomalies (list): List of detected anomalies.
    ax (Axes): Matplotlib Axes object to update the plot.
    """
    ax.clear()
    ax.scatter(np.arange(len(data)), data, s=5, label='Data Stream', color='blue')
    if anomalies:
        anomalies = np.array(anomalies)
        ax.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Anomalies')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Real-Time Data Stream with Anomalies')
    ax.legend()

def main():
    """
    Main function to simulate data stream, detect anomalies, and plot the results in real-time.
    """
    data_stream = simulate_data_stream()
    data = []
    anomalies = []
    fig, ax = plt.subplots(figsize=(15, 5))

    def update(frame):
        nonlocal data, anomalies
        for _ in range(100):  # Process 100 data points per update to speed things up
            data.append(next(data_stream))
        anomalies = detect_anomalies_streaming(data, window_size=100, threshold=0.005, batch_size=100)
        plot_anomalies(data, anomalies, ax)

    ani = animation.FuncAnimation(fig, update, interval=1000)  # Update the plot more frequently
    plt.show()

if __name__ == "__main__":
    main()
