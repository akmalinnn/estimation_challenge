import numpy as np
import matplotlib.pyplot as plt

def visualize_optical_flow(npy_file, step=10):
    """
    Visualize optical flow from a .npy file using quiver plot.
    
    Args:
        npy_file (str): Path to the .npy file containing optical flow.
        step (int): Step size to reduce the number of arrows for clarity.
    """
    # Load optical flow data
    flow = np.load(npy_file)  # Shape: (H, W, 2), where last dimension is (dx, dy)
    
    H, W, _ = flow.shape
    Y, X = np.mgrid[0:H:step, 0:W:step]  # Grid of points
    U, V = flow[::step, ::step, 0], flow[::step, ::step, 1]  # Flow vectors

    # Plot optical flow
    plt.figure(figsize=(10, 10))
    plt.imshow(np.zeros((H, W)), cmap='gray')  # Background
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=1)
    plt.title("Optical Flow Visualization")
    plt.show()


visualize_optical_flow("diff_flow_05_0012_frame_00155.png.npy")
