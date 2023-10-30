import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
from shapely.geometry import LineString


def calculate_angles(trajectory):
    vectors = np.diff(trajectory, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    cos_angles = np.einsum('ij,ij->i', vectors[:-1], vectors[1:]) / (norms[:-1] * norms[1:] + 1e-7)
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    return np.degrees(angles)


def find_feature_points(trajectory, theta=25):
    angles = calculate_angles(trajectory)
    # Feature points are where the change in direction is above the threshold
    feature_points = np.where(angles > theta)[0] + 1
    return feature_points


def find_loitering_start(trajectory, theta=25, window_size=5, feature_points_threshold=2):
    feature_points = find_feature_points(trajectory, theta)
    window = np.ones(window_size)
    counts = np.convolve(feature_points, window, 'valid')
    starts = np.where(counts >= feature_points_threshold)[0]
    return starts[0] if starts.size > 0 else None


def find_loitering_start_and_end(trajectory, theta=25, window_size=5, feature_points_threshold=2):
    feature_points = find_feature_points(trajectory, theta)
    start, end = None, None
    for i in range(len(feature_points) - window_size):
        if np.count_nonzero(feature_points[i:i + window_size]) >= feature_points_threshold:
            if start is None:
                start = feature_points[i]
            end = feature_points[i + window_size - 1]
    return start, end


def getMinEllipse(points, tol=0.001, max_iter=1000):
    n, d = points.shape
    Q = np.hstack((points, np.ones((n, 1))))
    Q = np.transpose(Q)
    err = 1.0
    u = np.ones(n) / n
    iter_count = 0

    # Khachiyan Algorithm
    while err > tol and iter_count < max_iter:
        X = np.dot(np.dot(Q, np.diag(u)), np.transpose(Q))
        M = np.diag(np.dot(np.dot(np.transpose(Q), np.linalg.inv(X)), Q))
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
        iter_count += 1

    # center of the ellipse
    center = np.dot(points.T, u)

    # shape matrix
    U = points - center  # [:, np.newaxis]
    # shape = np.dot(np.dot(U, np.diag(u)), U.T) / d
    shape = np.dot(np.dot(U.T, np.diag(u)), U) / d

    return center, shape


def ellipse_loitering_detection(coordinates, T0, T1, S_threshold, ID, display=False):
    if len(coordinates) <= T0:
        return False

    if T1 is None:
        T1 = len(coordinates)
    for i in range(T0, T1):
        points = coordinates[i:]
        if len(set(points[:, 0])) > 1 and len(set(points[:, 1])) > 1:
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = coordinates[hull.vertices]

                    if len(hull_points) < 3:
                        return False

                    center = np.mean(hull_points, axis=0)

                    polar_coords = np.array(
                        [
                            (
                                np.degrees(
                                    np.arctan2(
                                        point[1] - center[1], point[0] - center[0]
                                    )
                                ),
                                np.linalg.norm(point - center),
                            )
                            for point in hull_points
                        ]
                    )

                    sorted_indices = np.lexsort(
                        (polar_coords[:, 1], -polar_coords[:, 0])
                    )
                    sorted_points = hull_points[sorted_indices]

                    center, shape = getMinEllipse(sorted_points)

                    S = np.pi * np.sqrt(np.linalg.det(shape))

                    if S > S_threshold:
                        # Calculate the direction vector of the line from the start point to the end point
                        direction_vector = coordinates[T1] - coordinates[T0]
                        # Calculate the angle of this direction vector
                        overall_angle = 270 + np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
                        if display:
                            fig, ax = plt.subplots()
                            ax.plot(coordinates[:, 0], coordinates[:, 1], "b")
                            for simplex in hull.simplices:
                                plt.plot(points[simplex, 0], points[simplex, 1], 'r-')  # plot the convex hull

                            eigenvalues, eigenvectors = np.linalg.eigh(shape)
                            ellipse = Ellipse(
                                xy=center,
                                width=2 * np.sqrt(eigenvalues[0]),
                                height=2 * np.sqrt(eigenvalues[1]),
                                angle=overall_angle,  # Use the overall angle instead of the angle from the eigenvectors
                                fill=False,
                                color="r",
                            )
                            ax.add_artist(ellipse)
                            ax.set_title(f"Ellipse Loitering Detected ID: {ID}")
                            ax.invert_yaxis()
                            plt.xlim([0, 384])
                            plt.ylim([288, 0])
                            plt.gca().set_xticks([])  # Remove x-axis tick marks
                            plt.gca().set_yticks([])  # Remove y-axis tick marks
                            plt.savefig(f"plots/ellipse/{ID}_ellipse.pdf")
                            plt.close()
                        return True
                except:
                    return False
            return False


def convex_hull_loitering_detection(coordinates, T0, T1, S_threshold, ID, display=False):
    if len(coordinates) <= T0:
        return False

    if T1 is None:
        T1 = len(coordinates)

    for i in range(T0, T1):
        points = coordinates[i:]
        if len(set(points[:, 0])) > 1 and len(set(points[:, 1])) > 1:
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = coordinates[hull.vertices]

                    if len(hull_points) < 3:
                        return False

                    S = hull.area

                    if display:
                        fig, ax = plt.subplots()
                        ax.plot(coordinates[:, 0], coordinates[:, 1], "b")
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'r-')  # plot the convex hull

                        ax.set_title(f"Convex Hull Loitering Detected ID: {ID}")
                        ax.invert_yaxis()
                        plt.xlim([0, 384])
                        plt.ylim([288, 0])
                        plt.gca().set_xticks([])  # Remove x-axis tick marks
                        plt.gca().set_yticks([])  # Remove y-axis tick marks
                        plt.savefig(f"plots/convex_hull/{ID}_convex_hull.pdf")
                        plt.close()

                    if S > S_threshold:
                        return True

                except:
                    return False
            return False


def convex_hull_loitering_detection_2(coordinates, S_threshold, ID, display=False):
    # Check if we have enough coordinates to form a convex hull
    if len(coordinates) < 3:
        return False

    # Ensure there's more than one unique x and y coordinate
    if len(set(coordinates[:, 0])) <= 1 or len(set(coordinates[:, 1])) <= 1:
        return False

    try:
        hull = ConvexHull(coordinates)
        if display:
            fig, ax = plt.subplots()
            ax.plot(coordinates[:, 0], coordinates[:, 1], "b")
            for simplex in hull.simplices:
                plt.plot(coordinates[simplex, 0], coordinates[simplex, 1], 'r-')  # plot the convex hull

            ax.set_title(f"Convex Hull Loitering Detected ID: {ID}")
            ax.invert_yaxis()
            plt.xlim([0, 384])
            plt.ylim([288, 0])
            plt.gca().set_xticks([])  # Remove x-axis tick marks
            plt.gca().set_yticks([])  # Remove y-axis tick marks
            plt.savefig(f"plots/convex_hull/{ID}_convex_hull.pdf")
            plt.close()
        if hull.area > S_threshold:
            return True
    except:
        return False

    return False


class RectangleLoiteringDetection:
    def __init__(self, trajectory, theta=50, T0=60, S_threshold=1):
        self.trajectory = trajectory
        self.theta = np.deg2rad(theta)  # converting degrees to radians
        self.T0 = T0
        self.S_threshold = S_threshold
        self.feature_points = []
        self.current_vector = None

    def angle_between(self, v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude_product = np.linalg.norm(v1) * np.linalg.norm(
            v2) + 1e-7  # Add small constant to avoid division by zero
        ratio = dot_product / magnitude_product
        clipped_ratio = np.clip(ratio, -1, 1)  # Clip values to the range -1 to 1
        angle = np.arccos(clipped_ratio)
        return angle

    def calculate_vector(self, point1, point2):
        return np.array(point2) - np.array(point1)

    def detect_loitering(self):
        for i in range(1, len(self.trajectory)):
            if self.current_vector is None:
                self.current_vector = self.calculate_vector(self.trajectory[0], self.trajectory[i])
            else:
                new_vector = self.calculate_vector(self.trajectory[i - 1], self.trajectory[i])
                if self.angle_between(self.current_vector, new_vector) > self.theta:
                    self.feature_points.append(self.trajectory[i])
                    self.current_vector = new_vector

            if len(self.feature_points) > 4 and i > self.T0:
                x_coords = [point[0] for point in self.feature_points]
                y_coords = [point[1] for point in self.feature_points]
                S = abs(max(x_coords) - min(x_coords)) * abs(max(y_coords) - min(y_coords))
                if S > self.S_threshold:
                    return True  # loitering detected

        return False

    def plot_trajectory(self, ID):
        fig, ax = plt.subplots()

        # Plotting the entire trajectory
        x_coords = [point[0] for point in self.trajectory]
        y_coords = [point[1] for point in self.trajectory]
        ax.plot(x_coords, y_coords, label='Trajectory')

        # Plotting the feature points
        if self.feature_points:
            feature_x = [point[0] for point in self.feature_points]
            feature_y = [point[1] for point in self.feature_points]
            ax.scatter(feature_x, feature_y, color='red', label='Feature Points')

            # Drawing the rectangle if loitering is detected
            if len(self.feature_points) > 4:
                rect_x_min = min(feature_x)
                rect_y_min = min(feature_y)
                rect_width = max(feature_x) - rect_x_min
                rect_height = max(feature_y) - rect_y_min
                rectangle = patches.Rectangle((rect_x_min, rect_y_min), rect_width, rect_height, linewidth=1,
                                              edgecolor='r', facecolor='none')
                ax.add_patch(rectangle)
        ax.set_title(f"Rectangle Loitering Detected ID: {ID}")
        ax.invert_yaxis()
        plt.xlim([0, 384])
        plt.ylim([288, 0])
        plt.gca().set_xticks([])  # Remove x-axis tick marks
        plt.gca().set_yticks([])  # Remove y-axis tick marks
        ax.legend()
        plt.savefig(f"plots/rectangle/{ID}_rectangle.pdf")
        plt.close()
        # plt.show()


def closed_areas_loitering_detection(P, S_threshold=200, ID=None, display=False):
    # P = [(p[0], max(p[1] for p in P) - p[1]) for p in P]
    ellipses = []
    lines = [LineString([P[i], P[i + 1]]) for i in range(len(P) - 1)]

    for i, line1 in enumerate(lines[:-1]):
        for j, line2 in enumerate(lines[i + 2:], start=i + 2):
            if line1.intersects(line2):
                loop = P[i:j + 2]
                loop_center = np.mean(loop, axis=0)
                loop_dists = np.linalg.norm(loop - loop_center, axis=1)
                a, b = np.max(loop_dists), np.min(loop_dists)
                ellipse = Ellipse(loop_center, 2 * a, 2 * b, fill=False, edgecolor="r", linewidth=1)
                ellipses.append(ellipse)

    # if ellipses not empty
    if len(ellipses) > 0:
        # Get the ellipse with the largest area
        major_ellipse = max(ellipses, key=lambda e: e.width * e.height)
        area_major = major_ellipse.width * major_ellipse.height
        if ID is not None and display:
            fig, ax = plt.subplots()
            ax.plot(np.array(P)[:, 0], np.array(P)[:, 1], "b")
            ax.add_artist(major_ellipse)
            ax.set_title(f"Closed Areas Loitering Detected ID: {ID}")
            # ax.invert_yaxis()
            ax.set_xlim([0, 384])
            ax.set_ylim([288, 0])
            plt.savefig(f"plots/closed_areas/{ID}_closed_areas.pdf")
            plt.close()

        if area_major > S_threshold:
            return True
        else:
            return False
    else:
        return False


def no_motion_loitering_detection(trajectory, mode='short_term', frame_threshold=60, radius=5, std_threshold=0.1,
                                  ID=None, display=False):
    def plot_loitering(trajectory, center, radius, ID):
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1], "b")
        circle = plt.Circle(center, radius, fill=False, edgecolor="r", linewidth=1)
        plt.gca().add_artist(circle)
        plt.title(f"No Motion Loitering Detected ID: {ID}")
        plt.gca().invert_yaxis()
        plt.xlim([0, 384])
        plt.ylim([288, 0])
        plt.gca().set_xticks([])  # Remove x-axis tick marks
        plt.gca().set_yticks([])  # Remove y-axis tick marks
        plt.savefig(f"plots/no_motion/{ID}_no_motion.pdf")
        plt.close()

    if mode not in ['short_term', 'long_term']:
        print("Invalid mode. Choose 'short-term' or 'long-term'.")
        return False

    if len(trajectory) < 2:
        return False

    # Long-term loitering detection
    if mode == 'long_term':
        center = np.mean(trajectory, axis=0)
        distances = np.linalg.norm(trajectory - center, axis=1)
        std = np.std(distances)

        if display:
            plot_loitering(trajectory, center, radius, ID)

        if np.all(distances <= radius) and std <= std_threshold:
            return True

    # Short-term loitering detection
    elif mode == 'short_term':
        if frame_threshold <= 1:
            return False
        if frame_threshold >= len(trajectory):
            return False

        for i in np.arange(0, len(trajectory) - frame_threshold + 1,
                           frame_threshold // 2):  # step half the frame_threshold for overlap
            subset = trajectory[i:i + frame_threshold]
            center = subset.mean(axis=0)
            distances = np.linalg.norm(subset - center, axis=1)
            std = distances.std()

            if display:
                plot_loitering(trajectory, center, radius, ID)

            if np.all(distances <= radius) and std <= std_threshold:
                return True

    return False


def sector_loitering_detection(trajectory, T0=60, M1=5, M2=10, S_threshold=500, ID=None, display=False):
    P0 = trajectory[0]
    D_min, D_max = np.inf, -np.inf
    start_point = 0
    D_min_index, D_max_index = 0, 0

    for i in range(1, len(trajectory)):
        Pi = trajectory[i]
        Di = np.linalg.norm(Pi - P0)

        if i > T0:
            if Di < D_min:
                D_min = Di
                D_min_index = i
            if Di > D_max:
                D_max = Di
                D_max_index = i

        S = np.pi * (D_max ** 2 - D_min ** 2)

        if M1 < abs(D_max - D_min) < M2 and S > S_threshold:
            loitering_detected = True
            start_point = i - T0
            break
        else:
            loitering_detected = False

    if display and ID is not None:
        fig, ax = plt.subplots()

        ax.plot(trajectory[:, 0], trajectory[:, 1], "b")

        # Plot Dmin and Dmax vectors
        ax.quiver(trajectory[start_point, 0], trajectory[start_point, 1], trajectory[D_min_index, 0] - P0[0],
                  trajectory[D_min_index, 1] - P0[1], angles='xy', scale_units='xy', scale=1, color='g', label='Dmin')
        ax.quiver(trajectory[start_point, 0], trajectory[start_point, 1], trajectory[D_max_index, 0] - P0[0],
                  trajectory[D_max_index, 1] - P0[1], angles='xy', scale_units='xy', scale=1, color='r', label='Dmax')

        # Plot Dt vector (displacement from initial detection coordinate to current coordinate)
        ax.quiver(trajectory[start_point, 0], trajectory[start_point, 1], trajectory[-1, 0] - P0[0],
                  trajectory[-1, 1] - P0[1], angles='xy', scale_units='xy', scale=1, color='b', label='Dt')

        ax.legend()

        ax.set_title(f"Sector Loitering Detected ID: {ID}")
        ax.invert_yaxis()
        ax.set_xlim([0, 384])
        ax.set_ylim([288, 0])

        plt.savefig(f"plots/sector/{ID}_sector.pdf")
        plt.close()

    if loitering_detected:
        return loitering_detected
