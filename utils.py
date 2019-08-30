import cv2
import numpy as np
from scipy import signal
import torch


def optical_flow(I1g, I2g, window_size, tau=1e-2):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = round(window_size/2)  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    # I1g = I1g / 255.  # normalize pixels
    # I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0], 1))
            A = np.vstack((Ix, Iy)).T
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)
                u[i, j] = nu[0]
                v[i, j] = nu[1]

    return u, v


def optical_flow2(frame1, frame2):
    # vis = np.zeros((384, 836), np.float32)
    # h, w = vis.shape
    # vis2 = cv2.CreateMat(h, w, cv2.CV_32FC3)
    # vis0 = cv2.fromarray(vis)
    # cv2.CvtColor(vis0, vis2, cv2.CV_GRAY2BGR)

    # tmp = cv2.CreateMat(frame1.shape[0], frame1.shape[1], cv2.CV_32FC3)
    frame1 = cv2.fromarray(frame1)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # cv2.imshow('frame2', bgr)
    # cv2.imwrite('opticalfb.png', frame2)
    # cv2.imwrite('opticalhsv.png', bgr)

    return bgr


def pairwise_distances(x, y=None):
    """
    Code taken from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    Input: x is a Nxd matrix
           y is an optional Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


class Node:
    def __init__(self, point, cost, direction=None, parent=None):
        self.point = point
        self.cost = cost
        self.direction = direction
        self.parent = parent


def pathfinding(matrix):
    MAX_CONSECUTIVE_STEPS = 3
    print("Matrix shape", matrix.shape)

    starting_points = []
    # Find the starting points
    minimums = find_local_minimums(matrix[0].copy())
    for minimum in minimums:
        starting_points.append((0, minimum))
    minimums = find_local_minimums(matrix.T[0].copy())
    for minimum in minimums:
        starting_points.append((minimum, 0))
    print("Starting points", starting_points)

    end_nodes = []
    for starting_point in starting_points:
        current_node = Node(starting_point, 0)
        closed_nodes = {}
        opened_nodes = []

        # Iterate through the matrix while our current node has not reached an edge
        print(current_node.point)
        while current_node.point[0] < matrix.shape[0] - 1 and current_node.point[1] < matrix.shape[1] - 1:
            # If the current point has not already been explored or if it has a lower cost
            if current_node.point not in closed_nodes or current_node.cost < closed_nodes[current_node.point]:
                # Put the current node in the closed set
                closed_nodes[current_node.point] = current_node.cost
                # Prevent too many consecutive steps in the same direction
                consecutive_direction = current_node.direction
                consecutive_steps = 0
                if consecutive_direction != "DIAG":
                    previous_node = current_node.parent
                    while previous_node is not None and previous_node.direction == consecutive_direction and consecutive_steps < MAX_CONSECUTIVE_STEPS:
                        previous_node = current_node.parent
                        consecutive_steps += 1
                if consecutive_steps < MAX_CONSECUTIVE_STEPS:
                    consecutive_direction = None
                # Generate the neighbors
                neighbors = []
                if consecutive_direction != "DOWN":
                    neighbors.append(((current_node.point[0] + 1, current_node.point[1]), "DOWN"))
                if consecutive_direction != "RIGHT":
                    neighbors.append(((current_node.point[0], current_node.point[1] + 1), "RIGHT"))
                if consecutive_direction != "DIAG":
                    neighbors.append(((current_node.point[0] + 1, current_node.point[1] + 1), "DIAG"))
                # Add the neighbors to the opened list
                for neighbor in neighbors:
                    opened_nodes.append(Node(neighbor[0], current_node.cost + matrix[neighbor[0]], neighbor[1], current_node))
                # Sort the opened list to find the node with the lowest cost
                opened_nodes.sort(key=lambda x: x.cost)
            # Get the lowest cost node
            current_node = opened_nodes.pop(0)
        end_nodes.append(current_node)
    return end_nodes


def find_local_minimums(array):
    from matplotlib import pyplot as plt
    array = array - array.mean()
    moving_average = get_moving_average(array, 15)
    mins = []
    current_min = np.inf
    current_min_index = -1
    for index, (value, mov_avg) in enumerate(zip(array, moving_average)):
        if mov_avg > 0 and current_min < 0:
            mins.append((current_min_index, current_min))
            current_min = np.inf
        elif mov_avg < 0 and value < current_min:
            current_min = value
            current_min_index = index
    plt.plot(moving_average, color='gray')
    plt.plot(array)
    plt.axhline(0, color='black')
    for min in mins:
        plt.axvline(min[0], color='purple')
    plt.show()
    return [min[0] for min in mins]


def gaussian_kernel_1d(n, sigma=1):
    r = range(-int(n/2), int(n/2)+1)
    return [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r]


def get_moving_average(values, N):
    gaussian_kernel = np.array(gaussian_kernel_1d(N, sigma=4))
    moving_average = np.convolve(values, gaussian_kernel/gaussian_kernel.sum(), mode='same')
    return moving_average
