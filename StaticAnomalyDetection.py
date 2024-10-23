import numpy as np
import matplotlib.pyplot as plt

def simulate_data_stream(length=1000):
    """
    Simulates a data stream of floating-point numbers with seasonal patterns, noise, and regular patterns.

    Args:
    length (int): Number of data points to generate.

    Returns:
    np.ndarray: Simulated data stream.
    """
    if length <= 0:
        raise ValueError("Length of data stream must be positive.")
        
    np.random.seed(42)  # For reproducibility
    time = np.arange(length)  # Time index for the data stream
    seasonal_pattern = 10 * np.sin(2 * np.pi * time / 100)  # Simulated seasonal pattern
    noise = np.random.normal(0, 10, length)  # Simulated random noise with higher variance
    trend = 0.1 * time  # Simulated upward trend over time
    data = 50 + seasonal_pattern + noise + trend  # Combined data stream
    print('The shape of data is:', data.shape)
    return data

def detect_anomalies_streaming(data_stream, window_size=100, threshold=0.005):
    """
    Detects anomalies in a streaming data sequence using Gaussian Distribution.

    Args:
    data_stream (iterable): Continuous data stream.
    window_size (int): Number of data points in the sliding window.
    threshold (float): Threshold for anomaly detection based on PDF value.

    Returns:
    list: List of tuples containing the index and value of detected anomalies.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    if not isinstance(threshold, (float, int)) or threshold <= 0:
        raise ValueError("Threshold must be a positive number.")
    
    window = []  # Sliding window to hold recent data points
    anomalies = []  # List to store detected anomalies
    for i, point in enumerate(data_stream):
        if not isinstance(point, (float, int)):
            raise ValueError("Data stream values must be numeric.")
            
        window.append(point)
        if len(window) > window_size:
            window.pop(0)  # Maintain the window size
        
        if len(window) == window_size:
            mean = np.mean(window)  # Calculate mean of the window
            std_dev = np.std(window)  # Calculate standard deviation of the window
            
            if std_dev == 0:
                continue  # Skip if standard deviation is zero to avoid division by zero error
            
            pdf_value = (1 / (np.sqrt(2 * np.pi * (std_dev**2)))) * np.exp(-(point - mean)**2 / (2 * (std_dev)**2))  # Calculate PDF value
            
            if pdf_value < threshold:  # Check if the point is an anomaly
                anomalies.append((i, point))
                print(f"Anomaly detected at point {i}: {point}")
    
    return anomalies

def plot_anomalies(data, anomalies):
    """
    Plots the data stream and highlights detected anomalies.

    Args:
    data (iterable): Original data stream.
    anomalies (list): List of detected anomalies.
    """
    if not isinstance(data, (np.ndarray, list)):
        raise ValueError("Data must be a list or numpy array.")
    if not isinstance(anomalies, list):
        raise ValueError("Anomalies must be a list.")
    
    plt.figure(figsize=(15, 5))  # Set the size of the plot
    plt.scatter(np.arange(len(data)), data, label='Data Stream', s=5)  # Plot the data stream
    
    if anomalies:
        anomalies = np.array(anomalies)
        plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Anomalies')  # Highlight anomalies
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Simulation of Data Stream with Anomalies')
    plt.legend()
    plt.savefig('output.png')  # Save the plot as an image

def main():
    """
    Main function to simulate data stream, detect anomalies, and plot the results.
    """
    try:
        data_stream = simulate_data_stream()  # Generate the simulated data stream
        data = list(simulate_data_stream())  # For plotting purposes
        anomalies = detect_anomalies_streaming(data_stream)  # Detect anomalies in the data stream
        plot_anomalies(data, anomalies)  # Plot the data and anomalies
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
