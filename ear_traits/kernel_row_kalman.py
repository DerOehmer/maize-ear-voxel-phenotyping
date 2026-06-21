import numpy as np
import pandas as pd
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from collections import defaultdict
from scipy.stats import chi2
from dataclasses import dataclass, field

from ear_traits.configs import EarTraitConfigs
from ear_traits.scan_time_utils import CamExtrinsics


@dataclass
class KernelRow:
    kernel_ids: list[tuple] = field(default_factory=list)
    bottom_end: int | None = None
    top_end: int | None = None

    def __len__(self):
        return len(self.kernel_ids)


class KalmanFilter3D:
    def __init__(self, initial_point, initial_velocity, dt=1.0):
        """
        Initialize the Kalman filter for 3D tracking.
        --------------
        Parameters:
        initial_point : array-like
            Initial position in 3D space (x, y, z).
        initial_velocity : array-like
            Initial velocity in 3D space (vx, vy, vz).
        dt : float
            Time step for the state transition (default is 1.0).
        --------------

        """
        # State vector [x, y, z, vx, vy, vz]
        self.dt = dt
        self.x = np.array([*initial_point, *initial_velocity], dtype=float)

        # State covariance matrix
        self.P = np.eye(6) * 500.0

        # Process noise covariance
        q = 0.1
        self.Q = np.eye(6) * q

        # Observation noise covariance
        r = 1.0
        self.R = np.eye(3) * r

        # State transition matrix
        self.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Observation matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


class KernelRowAssignment:
    def __init__(self, kernel_df: pd.DataFrame, configs: EarTraitConfigs):
        self.kernel_df = kernel_df
        self.kernel_points = self.kernel_df[["x", "y", "z"]].values
        self.kernel_ids = self.kernel_df["kernel_id"].values
        # Use defaultdict to auto-create missing dicts and counters
        self.kernel_connections = defaultdict(lambda: defaultdict(float))

        self.avg_kernel_height = (
            (self.kernel_df["bbox_y2"] - self.kernel_df["bbox_y1"])
            / np.sqrt(self.kernel_df["px_to_mm2"])
        ).mean()
        self.configs = configs

    def predict_and_associate(
        self, kf: KalmanFilter3D, points: list[np.ndarray], gate_prob: float = 0.99
    ):
        """
        1) Predicts the next state with kf.predict()
        2) Computes the Mahalanobis distance from each detection to the predicted measurement
        3) Picks the detection with smallest distance within a chi2 gate
        4) Updates the filter with kf.update()

        Parameters
        ----------
        kf : KalmanFilter3D
            Your 3D Kalman filter instance.
        points : list of array-like, shape (3,)
            Candidate 3D measurements to associate.
        gate_prob : float
            Probability threshold for gating (default 0.99 for 3 DOF).

        Returns
        -------
        idx : int or None
            Index of the associated detection in `points`, or None if none passed the gate.
        maha_dist : float or None
            The Mahalanobis distance of the chosen detection, or None.
        """
        # 1) Predict step
        predicted_pos = kf.predict()  # advances kf.x and kf.P
        # Predicted measurement and its covariance
        z_pred = kf.H @ kf.x  # shape (3,)
        S = kf.H @ kf.P @ kf.H.T + kf.R  # innovation covariance
        S_inv = np.linalg.inv(S)

        # 2) Chi2 gating threshold for 3 degrees of freedom
        threshold = chi2.ppf(gate_prob, df=3)

        # 3) Find best match
        best_idx, best_d2 = None, np.inf
        for i, z in enumerate(points):
            y = z - z_pred
            d2 = float(y.T @ S_inv @ y)
            if d2 < best_d2:
                best_idx, best_d2 = i, d2

        matching_score = 1 - np.sqrt(best_d2) / np.sqrt(threshold)  # normalized score

        # 4) Check gate and update
        if best_d2 < threshold:
            kf.update(points[best_idx])
            return best_idx, matching_score
        else:
            return None, matching_score

    def nearest_point(self, predicted, points, threshold):

        distances = np.linalg.norm(points - predicted, axis=1)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        dist_score = 1 - min_distance / threshold
        if min_distance <= threshold:
            return points[min_idx], min_idx, dist_score
        else:
            return None, None, None

    def track_column(
        self,
        initial_kernel,
        upward_kernels,
        threshold=1.0,
        initial_velocity=np.array([0, 0, 0]),
    ):
        initial_point, initial_id = initial_kernel
        trajectory = [initial_id]
        traj_scores = []

        points, ids = upward_kernels
        points = np.array(points)

        kf = KalmanFilter3D(initial_point, initial_velocity)

        while True:
            idx, dist_score = self.predict_and_associate(kf, points)
            """predicted = kf.predict()
            match, idx, dist_score = nearest_point(predicted, points, threshold)
            
            kf.update(match)"""
            if idx is None:
                break
            trajectory.append(ids[idx])  # Store the ID of the matched point
            traj_scores.append(dist_score)  # Store the score of the match
            points = np.delete(points, idx, axis=0)  # Remove matched point
            # Remove the ID as well
            ids = np.delete(ids, idx)

        return trajectory, traj_scores

    def store_kernel_connections(self, kernels, scores, connections, upward=True):
        """
        Store kernel connections discovered by Kalman filter.
        --------------
        Parameters:
        kernels : list
            List of kernel IDs.
        connections : dict
            Dictionary mapping kernel IDs to their connected kernels.
        upward : bool
            If True, connections are stored in the direction of upward tracking.
        --------------
        """
        assert len(kernels) == len(set(kernels)), "Kernel IDs must be unique."
        if upward:
            # walk each consecutive pair (src → dst)
            for i, (src, dst) in enumerate(zip(kernels, kernels[1:])):
                connections[src][dst] += scores[i]
        else:
            for i, (dst, src) in enumerate(zip(kernels, kernels[1:])):
                connections[src][dst] += scores[i]

        # convert inner defaultdicts to plain dicts
        return connections

    def process_kernel_tracking(
        self,
        kernel_csv: pd.DataFrame,
        kernel_points: np.ndarray,
        kernel_ids: np.ndarray,
        kernel_connections: defaultdict[defaultdict[float]],
        avg_kernel_height: float,
        upward=True,
    ):
        """
        Processes each kernel in the kernel_csv to perform tracking and update kernel connections.

        Parameters:
        -----------
        kernel_csv : DataFrame
            DataFrame containing kernel traits.
        kernel_points : ndarray
            Array of kernel [x, y, z] points.
        kernel_ids : ndarray
            Array of kernel IDs.
        kernel_connections : defaultdict
            Dictionary mapping kernel IDs to their connected kernels.
        avg_kernel_height : float
            Average height of the kernels, used to estimate velocity.
        upward : bool
            If True, tracks upward; otherwise, tracks downward.

        Returns:
        --------
        kernel_connections : defaultdict
            Updated kernel connections after processing.
        """
        for kernel_id in kernel_csv["kernel_id"].values:
            initial_point = kernel_csv.loc[
                kernel_csv["kernel_id"] == kernel_id, ["x", "y", "z"]
            ].values[0]

            unit_vector = kernel_csv.loc[
                kernel_csv["kernel_id"] == kernel_id,
                ["unit_vector_x", "unit_vector_y", "unit_vector_z"],
            ].values[0]
            initial_velocity = unit_vector * avg_kernel_height

            if upward:
                # Rough velocity assumption
                """initial_velocity = np.array(
                    [0.0, 0.0, avg_kernel_height]
                )"""  # Adjust as necessary
                directional_mask = kernel_points[:, 2] > initial_point[2]
            else:
                """initial_velocity = np.array(
                    [0.0, 0.0, -avg_kernel_height]
                )"""  # Adjust as necessary
                initial_velocity = -initial_velocity
                directional_mask = kernel_points[:, 2] < initial_point[2]

            directional_points = kernel_points[directional_mask]
            directional_ids = kernel_ids[directional_mask]
            if len(directional_points) == 0:
                continue
            initial_kernel = (initial_point, kernel_id)
            directional_kernels = (directional_points, directional_ids)
            trajectory, traj_scores = self.track_column(
                initial_kernel, directional_kernels, avg_kernel_height, initial_velocity
            )
            # show_3d_points_plot(directional_points, directional_points, None)

            kernel_connections = self.store_kernel_connections(
                trajectory, traj_scores, kernel_connections, upward
            )

        return kernel_connections

    def prune_connections(self, connections: defaultdict[defaultdict[float]]):
        """
        Given a nested dict connections[src][dst] = count,
        return a new dict where for each src only the dst(s)
        with the maximum count are kept.
        """
        pruned = defaultdict(lambda: defaultdict(float))
        for src, dst_score in connections.items():
            if not dst_score:
                raise ValueError(f"No connections found for source {src}.")
            max_score = max(dst_score.values())
            # keep only the first dst with count == max_score (in case of ties)
            for dst, scr in dst_score.items():
                if scr == max_score:
                    pruned[src][dst] = scr
                    break
        return pruned

    def elev_azim_from_extrinsics(
        self,
        cam_extrinsics: CamExtrinsics,
        look_at: np.ndarray | None = None,
        points: np.ndarray | None = None,
    ):
        """
        cam_extrinsics: CamExtrinsics instance
        look_at: optional 3-array. If None and points provided, uses midpoint of their bounds.
        points: optional (N,3) array to auto-choose center if look_at not given.
        Returns (elev_deg, azim_deg)
        """
        C = cam_extrinsics.xyz.reshape(3)  # camera position in world coords

        if look_at is None:
            if points is not None:
                mins = points.min(axis=0)
                maxs = points.max(axis=0)
                look_at = 0.5 * (mins + maxs)
            else:
                look_at = np.zeros(3)

        v = C - look_at
        vx, vy, vz = v
        r_xy = np.hypot(vx, vy)
        elev = np.degrees(np.arctan2(vz, r_xy))
        azim = np.degrees(np.arctan2(vy, vx))
        return elev, azim

    def save_connected_3d_points_plot(
        self,
        output_file: str,
        cam_extrinsics: CamExtrinsics | None = None,
        pt_to_poly_in_view_d: np.ndarray | None = None,
        centre_poly_pts: np.ndarray | None = None,
        elev: float = 30,
        azim: float = 45,
        point_size: float = 7,
        dpi: int = 600,
        save: bool = True,
    ):
        """
        Saves a 3D scatter plot of points and their pruned connections to a file,
        with true data aspect ratio.

        Parameters:
        --------------
        output_file : str
            Where to save the image (png, pdf, svg, etc.).
        cam_extrinsics : CamExtrinsics | None
            Camera extrinsics to compute elevation and azimuth angles. If provided elev and azim are ignored.
        pt_to_poly_in_view_d : np.ndarray | None
            The difference of the distance from camera to polynom at point's z and from point to camera. Used to visualize occlusion in view.
        centre_poly_pts : np.ndarray | None
            (N,3) array of points along the center polynom to plot as dashed line
        elev, azim : float
            Viewing angles.
        point_size : float
            Marker size.
        dpi : int
            Output resolution.
        base_linewidth : float
            Minimum line width for connections.
        linewidth_scale : float
            Multiplier for line width per connection count.
        """

        if cam_extrinsics is not None:
            elev, azim = self.elev_azim_from_extrinsics(
                cam_extrinsics, points=self.kernel_points
            )

            # distances from points to camera
            pt_to_cam_dists = np.linalg.norm(
                self.kernel_points - cam_extrinsics.xyz.reshape(1, 3), axis=1
            )

        all_pts = np.asarray(self.kernel_points)  # ensure numpy array
        plt_kernel_ids = np.asarray(self.kernel_ids.copy())

        # Compute data ranges for aspect
        x_min, y_min, z_min = all_pts.min(axis=0)
        x_max, y_max, z_max = all_pts.max(axis=0)
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min

        if centre_poly_pts is None:
            # dummy line from (0,0,0) to (0,0,zmax +10)
            z_dummy = np.arange(0, z_max + 10, 1)
            x_dummy = np.zeros_like(z_dummy)
            y_dummy = np.zeros_like(z_dummy)
            centre_poly_pts = np.vstack((x_dummy, y_dummy, z_dummy)).T

        fig = plt.figure(figsize=(5, 6), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        # plot dashed line for centre poly
        ax.plot(
            centre_poly_pts[:, 0],
            centre_poly_pts[:, 1],
            centre_poly_pts[:, 2],
            c="red",
            linestyle="dashed",
            linewidth=0.5,
            zorder=-1,
            alpha=0.8,
            label="ear center poly",
        )

        pt_cmap = None
        d_to_ear_center = np.zeros(len(all_pts))

        if cam_extrinsics is not None:
            order = np.argsort(pt_to_cam_dists)
            d_to_ear_center = pt_to_poly_in_view_d[order]
            pt_cmap = "Purples_r"
            alpha_norm = plt.Normalize()

            all_pts = all_pts[order]
            plt_kernel_ids = plt_kernel_ids[order]
            # pt_alpha = np.where(pt_to_poly_in_view_d > 0, 0.3, 1.0)
            line_alpha = np.where(d_to_ear_center > 0, 0.3, 1.0)

        # Plot all points with per-point alpha. Matplotlib's `alpha` in
        # scatter is scalar-only, so build RGBA facecolors instead.
        if cam_extrinsics is not None:
            # scalar values aligned with all_pts (all_pts was reordered above)
            scalar_vals = pt_to_cam_dists[order]
            cmap_obj = plt.get_cmap(pt_cmap) if pt_cmap is not None else None
            if cmap_obj is not None:
                norm = plt.Normalize(vmin=np.min(scalar_vals), vmax=np.max(scalar_vals))
                colors = cmap_obj(norm(scalar_vals)).copy()  # Nx4 RGBA
            else:
                colors = np.ones((len(all_pts), 4), dtype=float)
                colors[:, :3] = 0.0  # black
            # set per-point alpha from line_alpha (already aligned to all_pts)
            colors[:, 3] = line_alpha
            edgecolors = np.zeros_like(colors)
            edgecolors[:, 3] = line_alpha
        else:
            # no camera info: opaque black points
            colors = np.ones((len(all_pts), 4), dtype=float)
            colors[:, :3] = 0.0
            colors[:, 3] = 1.0
            edgecolors = "k"

        ax.scatter(
            all_pts[:, 0],
            all_pts[:, 1],
            all_pts[:, 2],
            facecolors=colors,
            s=point_size,
            label="points",
            linewidths=0.1,
            edgecolors=edgecolors,
            depthshade=False,
            zorder=0,
        )

        # Draw each connection as a line
        # Setup colormap based on connection counts
        max_count = max(
            (cnt for dsts in self.kernel_connections.values() for cnt in dsts.values()),
            default=1,
        )
        max_krow_id = self.kernel_df["kernel_row_id"].max()

        # Draw connections
        for src, dsts in self.kernel_connections.items():
            src = int(src)
            src_idx = np.where(plt_kernel_ids == int(src))[0][0]
            p0 = all_pts[src_idx]
            for dst, count in dsts.items():
                dst = int(dst)
                dst_idx = np.where(plt_kernel_ids == dst)[0][0]
                p1 = all_pts[dst_idx]
                if d_to_ear_center[dst_idx] < 0:
                    line_zorder = 1
                else:
                    line_zorder = -1
                krow_id = self.kernel_df.loc[
                    self.kernel_df["kernel_id"] == src, "kernel_row_id"
                ].values[0]

                # Use the alpha corresponding to the source point in the reordered array
                line_alpha_val = (
                    line_alpha[src_idx] if cam_extrinsics is not None else 1.0
                )

                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    c=(0, 0, 0),
                    linewidth=0.3,
                    zorder=line_zorder,
                    alpha=line_alpha_val,
                )

        # Labels & view
        ax.grid(True)  # make sure grid is enabled

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0.25  # smaller value -> thinner lines
            axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.6)  # RGBA color + alpha

        # Add a colorbar to explain the colormap used for point coloring
        # (distance to camera). Only create if we have camera info and a cmap.

        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        # set_array can be the scalar values; required by colorbar API
        mappable.set_array(scalar_vals)
        # Make the colorbar smaller and slimmer. Use a smaller shrink and
        # a larger aspect to produce a tall, thin bar. Anchor it to the
        # right side with a small pad.
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            shrink=0.45,
            pad=-0.3,
            aspect=30,
            anchor=(0.5, 0.5),
        )
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Distance to Camera (mm)", fontsize=6)

        # reduce tick size
        ax.tick_params(axis="both", which="major", labelsize=6)
        # ax.tick_params(axis="both", which="minor", labelsize=2.5)

        plot_step = 20
        plot_xmin, plot_xmax = x_min - plot_step, x_max + plot_step
        plot_ymin, plot_ymax = y_min - plot_step, y_max + plot_step
        plot_zmin, plot_zmax = 0, z_max + plot_step
        x_plot_range = plot_xmax - plot_xmin
        y_plot_range = plot_ymax - plot_ymin
        z_plot_range = plot_zmax - plot_zmin

        # set limits with some padding
        abs_x_max = max(abs(plot_xmin), abs(plot_xmax))
        abs_y_max = max(abs(plot_ymin), abs(plot_ymax))
        abs_bot_max = max(abs_x_max, abs_y_max)
        ax.set_xlim(-abs_bot_max, abs_bot_max)
        ax.set_ylim(-abs_bot_max, abs_bot_max)
        ax.set_zlim(plot_zmin, plot_zmax)

        # set ticks for x y and z with plot_step mm steps
        bot_ticks = np.arange(
            int(-abs_bot_max + abs_bot_max % plot_step),
            int(abs_bot_max - abs_bot_max % plot_step) + 1,
            plot_step,
        )
        ax.set_xticks(bot_ticks)
        ax.set_yticks(bot_ticks)
        ax.set_zticks(np.arange(0, plot_zmax, plot_step))

        ax.view_init(elev=elev, azim=azim)

        # True aspect ratio
        ax.set_box_aspect((x_plot_range, y_plot_range, z_plot_range))
        ax.tick_params(axis="x", pad=-5)  # negative values bring ticks closer
        ax.tick_params(axis="y", pad=-5)
        ax.tick_params(axis="z", pad=-0.5)

        ax.set_xlabel("X", labelpad=-7)
        ax.set_ylabel("Y", labelpad=-7)
        ax.set_zlabel("Z", labelpad=-2)

        # Optional legend (only points)
        # ax.legend()

        # plt.tight_layout()
        if save:
            # Save the figure
            plt.savefig(output_file, dpi=dpi)
            plt.close(fig)
        else:
            # Show the plot
            plt.show()

    def invert_connections(self, connections: defaultdict[defaultdict[float]]):
        """
        Invert a dict[src][dst] = score into dict[dst][src] = score.
        If multiple sources map to the same dst, they all accumulate.
        """
        inverted = defaultdict(lambda: defaultdict(float))
        for src, dsts in connections.items():
            for dst, score in dsts.items():
                inverted[dst][src] = score
        return inverted

    def reorder_connected_by_z(
        self,
        connections: defaultdict[defaultdict[float]],
        points_xyz: np.ndarray,
        point_ids: np.ndarray,
    ):
        """
        connections: defaultdict(lambda: defaultdict(float))
            directed edges src→dst with a float score
        points_xyz: np.ndarray, shape (N,3)
        point_ids:  array of length N, giving the ID for each row in points_xyz

        Returns a new defaultdict(defaultdict(float)) containing, for each
        connected component, a single chain of edges linking the points in
        descending-Z order.
        """

        # 1) Map each ID to its Z coordinate
        id_to_z = {pid: z for pid, z in zip(point_ids, points_xyz[:, 2])}

        # 2) Build an undirected adjacency list to find components
        undirected = defaultdict(set)
        for src, inner in connections.items():
            for dst in inner:
                undirected[src].add(dst)
                undirected[dst].add(src)

        # 3) Find connected components via Depth-First Search (DFS)
        seen = set()
        components = []
        for node in undirected:
            if node not in seen:
                stack = [node]
                comp = set()
                while stack:
                    u = stack.pop()
                    if u in seen:
                        continue
                    seen.add(u)
                    comp.add(u)
                    for v in undirected[u]:
                        if v not in seen:
                            stack.append(v)
                components.append(comp)

        # 4) Rebuild each component as a sorted chain
        new_connections = defaultdict(lambda: defaultdict(float))
        for comp in components:
            # Sort this component's nodes by descending Z
            sorted_nodes = sorted(
                comp, key=lambda pid: id_to_z.get(pid, -np.inf), reverse=True
            )
            # Recreate edges between consecutive pairs
            for u, v in zip(sorted_nodes, sorted_nodes[1:]):
                # if original had the exact same directed edge, keep its score; else 1.0
                score = connections[u].get(v, 1.0)
                new_connections[u][v] = score

        return new_connections

    def get_kernel_row_chunks(
        self,
        connections: defaultdict[defaultdict[float]],
        kernel_points: np.ndarray,
        kernel_ids: np.ndarray,
        single_pt_ids: list[int] | None = None,
    ):
        """
        Sort connections by the Z coordinate of the source points.
        Returns a sorted list of tuples (src_id, dst_id, score).

        Parameters:
        -----------
        connections : defaultdict[defaultdict[int]]
            Dictionary mapping kernel IDs to their connected kernels.
        kernel_points : np.ndarray
            Array of kernel [x, y, z] points.
        kernel_ids : np.ndarray
            Array of kernel IDs.
        single_pt_ids : list[int] | None
            If provided, the list of single point IDs is treated as individual kernel rows.
            Else, single points are identified as those not in any connection.

        """
        sorted_connections = []
        # Ensure ordering by Z coordinate using the maximum of source and destination z values
        # Check 39320223226028 for bug
        # connections = check_connection_order(connections, kernel_points, kernel_ids)
        new_connections = self.reorder_connected_by_z(
            connections, kernel_points, kernel_ids
        )

        for src, dst in new_connections.items():
            src_idx = np.where(kernel_ids == src)[0][0]
            src_z = kernel_points[src_idx, 2]
            sorted_connections.append((src, list(dst.keys())[0], src_z))

        # Sort by source Z coordinate
        # sorted_connections.sort(key=lambda x: x[2], reverse=True)
        tracked_ids = []
        single_pts = []
        kernel_rows = []
        for connection in sorted_connections:
            src, dst, src_z = connection
            if src in tracked_ids:
                continue

            # Start a new kernel row
            kernel_row = KernelRow()
            kernel_row.top_end = src
            i = 0
            while src in new_connections.keys():
                kernel_row.kernel_ids.append(src)
                tracked_ids.append(src)
                src = list(new_connections[src].keys())[0]
                i += 1
                if i > 100:
                    raise RecursionError(
                        f"Too many iterations while tracking kernel row starting from {kernel_row.top_end}. "
                        "This may indicate a cycle in the connections."
                    )
            # also track final kernel of the row
            kernel_row.kernel_ids.append(src)
            tracked_ids.append(src)

            kernel_row.bottom_end = src
            kernel_rows.append(kernel_row)

        if not single_pt_ids:
            single_pts = [pt for pt in kernel_ids if pt not in tracked_ids]
            return kernel_rows, single_pts, new_connections
        else:
            for single_id in single_pt_ids:
                kernel_rows.append(
                    KernelRow(
                        kernel_ids=[single_id], bottom_end=single_id, top_end=single_id
                    )
                )
            return kernel_rows, None, new_connections

    def align_chunks_and_singles(
        self,
        kernel_rows: list[KernelRow],
        single_ids: list[int],
        pts_ids: tuple[np.ndarray, np.ndarray],
        kernel_height: float,
        kernel_connections: defaultdict[defaultdict[float]],
    ):
        """
        Aligns kernel rows with single points.
        Returns a list of kernel rows with single points added at the ends.

        Parameters:
        ----------
        kernel_rows : list[KernelRow]
            List of KernelRow objects, each representing a row of kernels.
        single_ids : list[int]
            List of single kernel IDs that are not part of any row.
        pts_ids : tuple[np.ndarray, np.ndarray]
            Tuple containing kernel points and their corresponding IDs.
        kernel_height : float
            Average height of the kernels, used to search for connections.
        kernel_connections : defaultdict[defaultdict[float]]
            Dictionary mapping kernel IDs to their connected kernels.
        """
        kernel_points, kernel_ids = pts_ids
        single_pts = kernel_points[np.isin(kernel_ids, single_ids)].tolist()

        bottom_ends = [
            kernel_points[np.where(kernel_ids == row.bottom_end)[0][0]]
            for row in kernel_rows
        ]
        top_ends = [
            kernel_points[np.where(kernel_ids == row.top_end)[0][0]]
            for row in kernel_rows
        ]

        kernel_row_skip_idcs = []

        for i, row in enumerate(kernel_rows):
            if i in kernel_row_skip_idcs:
                continue
            if row.bottom_end is None or row.top_end is None:
                raise ValueError(
                    "Kernel row must have both top and bottom ends defined."
                )
            # find a match for our bottom end
            bot_xyz = kernel_points[np.where(kernel_ids == row.bottom_end)[0][0]]
            bot_candidates = top_ends + single_pts
            top_ends_len = len(top_ends)
            bot_pred = bot_xyz - np.array(
                [0, 0, kernel_height]
            )  # Predict next point based on height
            match, match_idx, score = self.nearest_point(
                bot_pred, bot_candidates, 2 * kernel_height
            )
            if match is not None:
                # If match_idx is in the early idcs, it must be a top end of another row
                if match_idx < top_ends_len:
                    matchs_kernel_id = kernel_rows[match_idx].top_end
                    top_ends[match_idx] = [999.0, 999.0, 999.0]  # Mark as used
                    kernel_row_skip_idcs.append(match_idx)
                # Otherwise, it must be a single point
                else:
                    matchs_kernel_id = single_ids[match_idx - top_ends_len]
                    single_pts.pop(match_idx - top_ends_len)
                    single_ids.pop(match_idx - top_ends_len)
                bottom_ends[i] = [999.0, 999.0, 999.0]  # Mark as used
                kernel_connections[row.bottom_end][matchs_kernel_id] += score
            # find a match for our top end
            top_xyz = kernel_points[np.where(kernel_ids == row.top_end)[0][0]]
            top_candidates = bottom_ends + single_pts
            bot_ends_len = len(bottom_ends)
            top_pred = top_xyz + np.array(
                [0, 0, kernel_height]
            )  # Predict previous point
            match, match_idx, score = self.nearest_point(
                top_pred, top_candidates, kernel_height
            )
            if match is not None:
                # If match_idx is in the early idcs, it must be a bottom end of another row
                if match_idx < bot_ends_len:
                    matchs_kernel_id = kernel_rows[match_idx].bottom_end
                    bottom_ends[match_idx] = [999.0, 999.0, 999.0]  # Mark as used
                    kernel_row_skip_idcs.append(match_idx)

                # Otherwise, it must be a single point
                else:
                    matchs_kernel_id = single_ids[match_idx - bot_ends_len]
                    single_pts.pop(match_idx - bot_ends_len)
                    single_ids.pop(match_idx - bot_ends_len)
                top_ends[i] = [999.0, 999.0, 999.0]
                kernel_connections[matchs_kernel_id][row.top_end] += score

        return kernel_connections

    def merge_single_points(
        self,
        kernel_connections: defaultdict[defaultdict[float]],
        kernel_points: np.ndarray,
        kernel_ids: np.ndarray,
        kernel_height: float,
    ):
        """
        Merges remaining single points into the kernel connections.

        Parameters:
        -----------
        kernel_connections : defaultdict[defaultdict[int]]
            Dictionary mapping kernel IDs to their conncted kenel ID in downward direction.
        kernel_points : ndarray
            Array of kernel [x, y, z] points.
        kernel_ids : ndarray
            Array of kernel IDs.
        kernel_height : float
            Average height of the kernels, used to estimate velocity.

        """

        upward_connections = self.invert_connections(kernel_connections)
        connected_ids = set(kernel_connections.keys()) | set(upward_connections.keys())

        single_ids = kernel_ids[np.isin(kernel_ids, list(connected_ids), invert=True)]

        remaining_singles = []

        for sid in single_ids:
            single_idx = np.where(kernel_ids == sid)[0][0]
            single_point = kernel_points[single_idx]
            # Remove the single point itself
            candidate_points = np.delete(kernel_points, single_idx, axis=0)

            # Remove the single point ID
            candidate_ids = np.delete(kernel_ids, single_idx, axis=0)
            match, match_idx, score = self.nearest_point(
                single_point, candidate_points, kernel_height * 2
            )
            if match is not None:
                match_id = candidate_ids[match_idx]
                if single_point[2] < match[2]:
                    if match_id in kernel_connections.keys():
                        next_id = list(kernel_connections[match_id].items())[0][0]
                        kernel_connections[sid][next_id] += score
                        kernel_connections[match_id].pop(next_id)

                    kernel_connections[match_id][sid] += score

                else:
                    if match_id in upward_connections.keys():
                        prev_id = list(upward_connections[match_id].items())[0][0]
                        kernel_connections[prev_id][sid] += score
                        kernel_connections[prev_id].pop(match_id)
                    kernel_connections[sid][match_id] += score

                upward_connections = self.invert_connections(kernel_connections)
            else:
                # If no match found, store the single point for later processing
                remaining_singles.append(sid)

        return kernel_connections, remaining_singles

    def assign_kernel_row_ids(self, kernel_rows: list[KernelRow]) -> pd.DataFrame:
        """
        Assigns unique IDs to each kernel row and creates a DataFrame for easy access.
        Each row is represented by its kernel IDs, bottom end, top end, and whether it contains a single kernel.

        Parameters:
        ----------
        kernel_rows : list[KernelRow]
            List of KernelRow objects, each representing a row of kernels.
        Returns:
        -------
        pd.DataFrame
            DataFrame containing the kernel row IDs, kernel IDs, bottom end, top end, and single kernel status.
        """
        df_rows = []
        for i, krow in enumerate(kernel_rows):
            for kernel_id in krow.kernel_ids:
                df_row = {
                    "kernel_row_id": i,
                    "kernel_id": int(kernel_id),
                    "row_bottom_end": kernel_id == krow.bottom_end,
                    "row_top_end": kernel_id == krow.top_end,
                    "single_kernel": len(krow.kernel_ids) == 1,
                }
                df_rows.append(df_row)
        return pd.DataFrame(df_rows)

    def reasign_krow_ids_by_polar_coordinates(
        self, kernel_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Reassign kernel row IDs based on polar coordinates to ensure spatial coherence.
        Kernels that are close in polar coordinates are grouped into the same kernel row.

        Parameters:
        ----------
        kernel_df : pd.DataFrame
            DataFrame containing kernel traits including 'kernel_id', 'x', 'y', 'z', and 'kernel_row_id'.

        Returns:
        -------
        pd.DataFrame
            Updated DataFrame with reassigned 'kernel_row_id'.
        """
        if "kernel_row_id" not in kernel_df.columns:
            raise ValueError("Input DataFrame must contain 'kernel_row_id' column.")

        # We will reassign kernel_row_id based on the polar coordinates of the
        # first kernel in each row (order_in_krow == 0). This ensures the row
        # labeling follows angular order around the ear.

        if "order_in_krow" not in kernel_df.columns:
            raise ValueError("Input DataFrame must contain 'order_in_krow' column.")

        # Select the first kernel (order_in_krow == 0) for each kernel_row
        first_kernels = (
            kernel_df.loc[kernel_df["order_in_krow"] == 0]
            .loc[:, ["kernel_row_id", "kernel_id", "x", "y"]]
            .dropna()
            .copy()
        )

        if first_kernels.empty:
            # Nothing to reassign — return original
            return kernel_df

        # Compute polar coords for those first kernels
        first_kernels["r"] = np.hypot(first_kernels["x"], first_kernels["y"])
        first_kernels["theta"] = np.arctan2(
            first_kernels["y"], first_kernels["x"]
        )  # [-pi, pi]

        # Normalize theta to [0, 2pi) for stable sorting
        first_kernels["theta_norm"] = (first_kernels["theta"] + 2 * np.pi) % (2 * np.pi)

        # Sort rows by angle (theta) primarily, radius secondarily
        first_kernels = first_kernels.sort_values(by=["theta_norm", "r"]).reset_index(
            drop=True
        )

        # Build mapping from old kernel_row_id -> new sequential id
        unique_old_krow_ids = first_kernels["kernel_row_id"].tolist()
        krow_id_map = {old: new for new, old in enumerate(unique_old_krow_ids)}

        # Apply mapping to a copy of the full DataFrame
        out_df = kernel_df.copy()
        out_df["kernel_row_id"] = out_df["kernel_row_id"].map(
            lambda v: krow_id_map.get(v, v)
        )

        assert out_df["kernel_row_id"].unique().size == len(
            unique_old_krow_ids
        ), "Kernel row ID reassignment failed."
        return out_df

    def run(self):

        # Process upward and downward tracking
        self.kernel_connections = self.process_kernel_tracking(
            self.kernel_df,
            self.kernel_points,
            self.kernel_ids,
            self.kernel_connections,
            self.avg_kernel_height,
            upward=True,
        )
        self.kernel_connections = self.process_kernel_tracking(
            self.kernel_df,
            self.kernel_points,
            self.kernel_ids,
            self.kernel_connections,
            self.avg_kernel_height,
            upward=False,
        )
        # Prune connections so no kernel has more than one upward and one downward connection
        pruned_connections = self.prune_connections(self.kernel_connections)
        inverted_connections = self.invert_connections(pruned_connections)
        bipruned_connections = self.prune_connections(inverted_connections)

        # Identify connections and connect possibly interrupted kernel rows
        kernel_rows, single_pts, z_ordered_connections = self.get_kernel_row_chunks(
            bipruned_connections, self.kernel_points, self.kernel_ids
        )
        alligned_connections = self.align_chunks_and_singles(
            kernel_rows,
            single_pts,
            (self.kernel_points, self.kernel_ids),
            self.avg_kernel_height,
            z_ordered_connections,
        )
        sp_merged_connections, remaining_single_points = self.merge_single_points(
            alligned_connections,
            self.kernel_points,
            self.kernel_ids,
            self.avg_kernel_height,
        )

        final_kernel_rows, _, self.kernel_connections = self.get_kernel_row_chunks(
            sp_merged_connections,
            self.kernel_points,
            self.kernel_ids,
            single_pt_ids=remaining_single_points,
        )
        # Assign unique IDs to each identified kernel row
        krow_df = self.assign_kernel_row_ids(final_kernel_rows)
        krow_df["order_in_krow"] = krow_df.groupby("kernel_row_id").cumcount()
        assert len(krow_df) == len(
            self.kernel_points
        ), "Kernel row DataFrame length does not match kernel points length."

        if len(krow_df) > 0:
            if "kernel_row_id" in self.kernel_df.columns:
                self.kernel_df = self.kernel_df.drop(
                    columns=["kernel_row_id", "order_in_krow"]
                )

            self.kernel_df = self.kernel_df.merge(
                krow_df.loc[:, ["kernel_id", "kernel_row_id", "order_in_krow"]],
                on="kernel_id",
                how="left",
            )
            # self.kernel_df = self.reasign_krow_ids_by_polar_coordinates(self.kernel_df)

        max_column_len = krow_df.groupby("kernel_row_id").count()["kernel_id"].max()
        return self.kernel_df, max_column_len


def main():
    import pickle
    import json

    kernel_csv = pd.read_csv("Results/_test(70)/_test_KernelTraits.csv")
    # read preprocessed cam extrinsics
    with open("Results/_test(70)/_test_cam_extrinsics.pkl", "rb") as f:
        cam_extrinsics = pickle.load(f)
    # remove kernel_row_id column for testing purposes
    if "kernel_row_id" in kernel_csv.columns:
        kernel_csv = kernel_csv.drop(columns=["kernel_row_id"])
    krow_assignment = KernelRowAssignment(
        kernel_csv,
        EarTraitConfigs(
            exp_name="test",
            reference=False,
            do_voxel_carving=True,
            do_kernel_count=True,
            do_sam_seg=True,
            safe_results=True,
            tracking_vis=True,
            hsv_thresh_p="test",
            archive_imgs=False,
            jetson=False,
            mm_per_voxelside=2,
            yolo_engine=None,
            debug=False,
            krow_assignment=True,
        ),
    )
    # Uncomment to run the full assignment
    ## kernel_df, max_column_len = krow_assignment.run()

    pt_to_poly_in_view_d = np.load("Results/_test(70)/pt_to_poly_in_view_d.npy")

    # Load previously computed results for testing visualization and save some processing time
    finished_kernel_df = pd.read_csv("Results/_test(70)/finished_kernel_df.csv")

    with open("Results/_test(70)/_test_kernel_points.pkl", "rb") as f:
        pickled_pts = pickle.load(f)

    with open("Results/_test(70)/_test_kernel_connectionsl.json", "r") as f:
        processed_connections = json.load(f)

    krow_assignment.kernel_df = finished_kernel_df
    krow_assignment.kernel_points = pickled_pts
    krow_assignment.kernel_connections = processed_connections

    krow_assignment.save_connected_3d_points_plot(
        "_test_kernel_rows.png",
        cam_extrinsics=cam_extrinsics,
        pt_to_poly_in_view_d=pt_to_poly_in_view_d,
    )
    print("debugging figure saved to _test_kernel_rows.png")


if __name__ == "__main__":
    main()
