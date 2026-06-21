import os
import json
from typing import List, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from ear_traits.scan_time_utils import CamIntrinsics, CamExtrinsics
from ear_traits.configs import EarTraitConfigs

import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sysconfig
import sys


def check_jetson():
    if "-aarch64-" in sysconfig.get_config_var("EXT_SUFFIX") or (
        sys.version_info.major == 3 and sys.version_info.minor == 8
    ):
        print("### Running on Jetson ###")
        return True
    print("### Running on standard machine ###")
    return False


GPU_DEVICE = "cpu" if check_jetson() else "gpu"


@dataclass
class TrackData:
    img_id: int
    tracked_kernel_xyz: List
    tracked_kernel_xyz_with_duplicates: List
    valid_bbox_indices: List
    valid_bbox_indices_with_duplicates: List
    tracked_scales: List


class VoxelCarvingJAX:
    def __init__(self, configs: EarTraitConfigs, voxel_grid_dims: tuple):
        self.mm_per_voxelside = (
            configs.mm_per_voxelside
        )  # 0.5 is consuming around 64 GB RAM
        self.voxel_shape, self.voxel_bounds = self.def_voxel_grid(
            self.mm_per_voxelside, voxel_grid_dims
        )

        self.hsv_thresh = self.load_hsv_thresh(configs.hsv_thresh_p)
        self.valid_voxel_coords = None
        self.projection_matrices = None
        self.poly_coeffs_x = None
        self.poly_coeffs_y = None
        self.max_matching_distance = None

        self.valid_mask_low = None
        self.valid_mask_up = None
        self.valid_masks = None

    def load_hsv_thresh(self, path):
        try:
            with open(path, "r") as f:
                hsv = json.load(f)
        except Exception:
            raise FileNotFoundError("HSV Threshholds for background not found")
        return (
            hsv["hmin"],
            hsv["hmax"],
            hsv["smin"] - 10,  # less strict
            hsv["smax"],
            hsv["vmin"],
            hsv["vmax"],
        )

    def def_voxel_grid(self, voxelside_per_mm, voxel_grid_dims):
        if voxel_grid_dims is None:
            voxel_shape = (
                int(100 / voxelside_per_mm),
                int(100 / voxelside_per_mm),
                int(300 / voxelside_per_mm),
            )  # resolution along x, y, z
            voxel_bounds = (
                (-50.0, 50.0),  # x-range in world coordinates
                (-50.0, 50.0),  # y-range
                (0.0, 300.0),  # 300 mm is the maximum ear height
            )
        else:
            x_bounds = voxel_grid_dims[0]
            y_bounds = voxel_grid_dims[1]
            z_bounds = voxel_grid_dims[2]
            voxel_shape = (
                int((x_bounds[1] - x_bounds[0]) / voxelside_per_mm),
                int((y_bounds[1] - y_bounds[0]) / voxelside_per_mm),
                int((z_bounds[1] - z_bounds[0]) / voxelside_per_mm),
            )
            voxel_bounds = voxel_grid_dims
        return voxel_shape, voxel_bounds

    def get_ear_volume(
        self,
        imgs: Tuple[List, List],
        extrinsics: Tuple[List, List],
        intrinsics: Tuple[CamIntrinsics, CamIntrinsics],
    ):
        # Convert the numpy mask (bool) to uint8 and then to a JAX array.
        self.low_cam_extrinsics = extrinsics[0]
        silhouettes, self.projection_matrices = self.preprocessing(
            imgs, extrinsics, intrinsics
        )
        occupancy_grid, voxel_coords = self.voxel_carving(
            self.voxel_shape,
            self.voxel_bounds,
            self.projection_matrices[self.valid_masks],
            silhouettes[self.valid_masks],
        )
        # Count only the voxels that survived carving.
        carved_voxel_count = occupancy_grid.sum()
        occupancy = occupancy_grid.reshape(-1)
        self.valid_voxel_coords = jax.device_get(voxel_coords[occupancy])

        # coefs = self.fit_ear_center(self.valid_voxel_coords)
        # print("Fitted coeffs", coefs)
        # white_coords_low = jnp.stack(jnp.nonzero(silhouettes[0]), axis=-1)  # shape: (P, 2)
        # dist_map_low = self.projection_based_distance_map(projection_matrices[0], self.valid_voxel_coords, white_coords_low,silhouettes[0].shape, jnp.array(extrinsics[0][0].xyz).flatten(), coefs)
        # self.dmap2(dist_map_low, "low")
        # white_coords_up = jnp.stack(jnp.nonzero(silhouettes[50]), axis=-1)  # shape: (P, 2)
        # dist_map_up = self.projection_based_distance_map(projection_matrices[50], self.valid_voxel_coords, white_coords_up, silhouettes[50].shape, jnp.array(extrinsics[1][0].xyz).flatten(), coefs)
        # self.dmap2(dist_map_up, "up")

        return carved_voxel_count * self.mm_per_voxelside**3 / 1000  # in cm^3/ml

    def get_low_img_center_polynoms(self, num_samples=12):
        """
        Process the voxel data and projection matrices to obtain a polynomial
        representation of the projected curve in each image.

        Parameters:
            degree (int): Degree of the 2D polynomial fit for the image curve.
            num_samples (int): Number of points to sample along the curve.

        Returns:
            image_pts (list): List of tuples (u, v) representing the sampled points.
        """
        low_cam_pms = self.projection_matrices[:50]
        z_max = jnp.amax(self.valid_voxel_coords[:, 2] + 3)  # 3mm buffer
        # Fit the 3D polynomial from voxel data
        self.poly_coeffs_x, self.poly_coeffs_y = self.fit_ear_center(
            self.valid_voxel_coords
        )
        image_pts = []

        for i, P in enumerate(low_cam_pms):
            # Define the t_range for sampling the polynomial curve.
            t_range = (-3, z_max)
            # Sample t values along the curve.
            t_values = np.linspace(t_range[0], t_range[1], num_samples)
            # Compute the corresponding 3D points on the fitted polynomial.
            x_values = (
                self.poly_coeffs_x[0] * t_values**2
                + self.poly_coeffs_x[1] * t_values
                + self.poly_coeffs_x[2]
            )
            y_values = (
                self.poly_coeffs_y[0] * t_values**2
                + self.poly_coeffs_y[1] * t_values
                + self.poly_coeffs_y[2]
            )
            pts_3d = np.stack([x_values, y_values, t_values], axis=-1)
            # Project the 3D curve onto the image using the projection matrix P.
            u, v = self.project_polynomial_to_image(
                self.poly_coeffs_x, self.poly_coeffs_y, P, t_range, num_samples
            )
            image_pts.append((u, v))
        return np.array(image_pts), np.array(pts_3d)

    def unit_direction_vector_at_z(self, z):
        """
        Compute the unit tangent vector of the fitted 3D polynomial curve through the ear.

        Args:
            coeffs_x: jnp.array of shape (3,), coefficients [a, b, c] for x(z) = a*z^2 + b*z + c
            coeffs_y: jnp.array of shape (3,), coefficients [d, e, f] for y(z) = d*z^2 + e*z + f
            z: float or jnp scalar, the height at which to compute the direction vector

        Returns:
            jnp.array of shape (3,), the unit direction vector [dx/dz, dy/dz, 1] normalized
        """
        a, b, _ = self.poly_coeffs_x
        d, e, _ = self.poly_coeffs_y

        dx_dz = 2 * a * z + b
        dy_dz = 2 * d * z + e
        dz_dz = 1.0

        direction = jnp.array([dx_dz, dy_dz, dz_dz])
        norm = jnp.linalg.norm(direction)

        return direction / norm

    def _append_empty_track_data(
        self, tracked_data_lst: List[TrackData], i: int
    ) -> List[TrackData]:
        tracked_data_lst.append(
            TrackData(
                img_id=i,
                tracked_kernel_xyz=[],
                tracked_kernel_xyz_with_duplicates=[],
                valid_bbox_indices=[],
                valid_bbox_indices_with_duplicates=[],
                tracked_scales=[],
            )
        )
        return tracked_data_lst

    def get_unseen_kernel_xyz(
        self, img_coordinates: List[List[List]], extrinsics: List[CamExtrinsics]
    ) -> Tuple[List[TrackData], List[str]]:
        self.valid_voxel_coords = jax.device_put(
            self.valid_voxel_coords, device=jax.devices(GPU_DEVICE)[0]
        )
        low_cam_pms = self.projection_matrices[:50]
        assert len(low_cam_pms) == 50

        tracked_data_lst: List[TrackData] = []
        failed_kernel_msgs: List[str] = []

        for img_id, (coords_img, extr_img, low_cam_pm) in enumerate(
            zip(img_coordinates, extrinsics, low_cam_pms)
        ):
            # Check image validity
            if not self.valid_mask_low[img_id]:
                tracked_data_lst = self._append_empty_track_data(
                    tracked_data_lst, img_id
                )
                continue
            # Check if the image has any coordinates
            if len(coords_img) == 0:
                tracked_data_lst = self._append_empty_track_data(
                    tracked_data_lst, img_id
                )
                continue

            min_pixel_distance_idcs = self.match_img_coords_to_voxel_coords(
                low_cam_pm,
                self.valid_voxel_coords,
                jnp.array(coords_img),
                jnp.array(extr_img.xyz).flatten(),
                self.max_matching_distance,
            )

            voxel_coords = self.valid_voxel_coords[
                min_pixel_distance_idcs[min_pixel_distance_idcs != -1]
            ]

            if len(voxel_coords) == 0:
                tracked_data_lst = self._append_empty_track_data(
                    tracked_data_lst, img_id
                )
                continue

            # select voxel inside kernel instead of form the surface
            voxel_coords = jnp.array(
                [
                    self.get_inside_kernel(jnp.array(extr_img.xyz).flatten(), xyz)
                    for xyz in voxel_coords
                ]
            )

            if len(min_pixel_distance_idcs[min_pixel_distance_idcs == -1]) > 0:
                msg = f"No voxel found for {len(min_pixel_distance_idcs[min_pixel_distance_idcs == -1])} kernel(s) in low image {img_id}"
                failed_kernel_msgs.append(msg)
                print(msg)

            missing_voxel_idcs = jnp.where(min_pixel_distance_idcs == -1)[0]
            flattened_tracked_kernel_xyz = [
                xyz
                for tracked_data in tracked_data_lst
                for xyz in tracked_data.tracked_kernel_xyz
            ]
            if len(flattened_tracked_kernel_xyz) == 0:
                unseen_kernel_xyz = voxel_coords
                valid_voxel_idcs = jnp.where(min_pixel_distance_idcs != -1)[0]
                indices = valid_voxel_idcs
                indices_with_duplicates = valid_voxel_idcs
            else:
                # Put less empahsis on x and y
                new_coords = voxel_coords.at[:, :2].set(voxel_coords[:, :2] * 0.5)
                prev_coords = jnp.array(flattened_tracked_kernel_xyz)
                prev_coords = prev_coords.at[:, :2].set(prev_coords[:, :2] * 0.5)
                _all_coord_dist, filter_mask = self.compute_distances_and_filter(
                    new_coords, prev_coords, 2.0
                )

                # Use the mask to filter
                complete_filter_mask = filter_mask.copy()
                # add previously filtered indices where no voxel was found
                for vidc in missing_voxel_idcs:
                    complete_filter_mask = jnp.insert(complete_filter_mask, vidc, False)
                assert len(complete_filter_mask) == len(coords_img)
                unseen_kernel_xyz = voxel_coords[filter_mask]
                indices = jnp.nonzero(complete_filter_mask)[0]
                indices_with_duplicates = jnp.where(min_pixel_distance_idcs != -1)[0]

            # Get pixel to mm scale
            px_to_mm2_scales = self.compute_px_per_mm2_scales(
                jnp.array(unseen_kernel_xyz), low_cam_pm
            )
            assert len(indices) == len(unseen_kernel_xyz) == len(px_to_mm2_scales)
            assert len(indices_with_duplicates) == len(voxel_coords)
            tracked_data_lst.append(
                TrackData(
                    img_id=img_id,
                    tracked_kernel_xyz=unseen_kernel_xyz,
                    tracked_kernel_xyz_with_duplicates=voxel_coords,
                    valid_bbox_indices=indices,
                    valid_bbox_indices_with_duplicates=indices_with_duplicates,
                    tracked_scales=px_to_mm2_scales,
                )
            )

        return tracked_data_lst, failed_kernel_msgs

    def get_low_img_tracked_kernel_xy(self, tracked_pts, kernel_ids, nth=1):
        low_cam_pms = self.projection_matrices[:50]
        tracked_pts = jnp.array(tracked_pts)
        proj_points = []
        point_ids = []
        kernels_in_view_ds = []
        for i, pm in enumerate(low_cam_pms):
            if len(tracked_pts) == 0 or i % nth != 0 or not self.valid_mask_low[i]:
                proj_points.append([])
                point_ids.append([])
                kernels_in_view_ds.append([])
                continue
            camera_position = jnp.array(self.low_cam_extrinsics[i].xyz).flatten()
            proj_points_masked, valid_mask, kernels_in_view_d = (
                self.project_xyz_on_images(
                    tracked_pts,
                    self.poly_coeffs_x,
                    self.poly_coeffs_y,
                    camera_position,
                    pm,
                )
            )
            if len(proj_points_masked[valid_mask]) == 0:
                proj_points.append([])
                point_ids.append([])
                kernels_in_view_ds.append([])
            else:
                proj_points.append(proj_points_masked[valid_mask])
                point_ids.append(np.array(kernel_ids)[valid_mask])
                kernels_in_view_ds.append(kernels_in_view_d)
        return proj_points, point_ids, kernels_in_view_ds

    def _get_max_matching_distance(self, intr: CamIntrinsics, emp_denom=1007769):
        """
        Compute the maximum orthogonal distance on the image plane
        to match a pixel to a voxel based on the camera intrinsics.

        Args:
            intr: Camera intrinsics object.
            emp_denom: Empirical denominator to calculate the maximum matching distance based on camera resolution.

        Returns:
            max_matching_distance: Maximum matching distance.
        """
        n_pixels = intr.w * intr.h
        return n_pixels / emp_denom * self.mm_per_voxelside

    def preprocessing(
        self,
        imgs: Tuple[List, List],
        extrinsics: Tuple[List, List],
        intrinsics: Tuple[CamIntrinsics, CamIntrinsics],
    ):
        self.max_matching_distance = self._get_max_matching_distance(intrinsics[0])
        low_imgs, up_imgs = imgs
        low_masks = np.array(
            [self.background_masking(low_img.img) for low_img in low_imgs]
        )
        low_inlier_mask = np.array([not img_obj.error_msg for img_obj in low_imgs])
        low_masks = low_masks

        self.mask_shape = low_masks[0].shape
        up_masks = np.array([self.background_masking(up_img.img) for up_img in up_imgs])
        up_inlier_mask = np.array([not img_obj.error_msg for img_obj in up_imgs])
        up_masks = up_masks
        masks = np.concatenate([low_masks, up_masks], axis=0)
        silhouettes = jnp.array(
            masks
        )  # Stack masks into a single JAX array of shape (m, h, w)

        low_intrinsic, up_intrinsic = intrinsics
        low_extrinsics, up_extrinsics = extrinsics
        low_projections = jnp.array(
            [
                self.compute_projection_matrix(low_intrinsic, lextr)
                for lextr in low_extrinsics
            ]
        )
        up_projections = jnp.array(
            [
                self.compute_projection_matrix(up_intrinsic, uextr)
                for uextr in up_extrinsics
            ]
        )
        projection_matrices = jnp.concatenate([low_projections, up_projections], axis=0)
        self.valid_mask_low = low_inlier_mask
        self.valid_mask_up = up_inlier_mask
        self.valid_masks = np.concatenate([low_inlier_mask, up_inlier_mask], axis=0)
        return silhouettes, projection_matrices

    def background_masking(self, img: np.ndarray):
        hmin, hmax, smin, smax, vmin, vmax = self.hsv_thresh
        imgBLUR = cv2.GaussianBlur(img, (3, 3), 0)
        imgHSV = cv2.cvtColor(imgBLUR, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([hmin, smin, vmin])
        higher_hsv = np.array([hmax, smax, vmax])
        mask = cv2.inRange(imgHSV, lower_hsv, higher_hsv)
        return mask.astype(np.uint8)

    def get_inside_kernel(self, cam_pos, kernel_surface_coord, depth=1.0):
        v = kernel_surface_coord - cam_pos
        d = jnp.linalg.norm(v)
        if d == 0:
            raise ValueError(
                "Point A and point B are identical; cannot determine direction."
            )

        # New distance from cam_pos to pt inside kernel
        new_distance = d + depth

        # Calculate the new point along the same direction
        inside_kernel_xyz = cam_pos + (new_distance / d) * v
        return inside_kernel_xyz

    def get_intr_3x3_jnp(self, intr: CamIntrinsics):
        return jnp.array(intr.undist_cam_mtx, dtype=jnp.float32)

    def get_extr_4x4_jnp(self, extr: CamExtrinsics):

        cam_center = extr.xyz  # already computed as -R^T @ t
        R_cw = extr.rotation_matrix.T  # camera-to-world rotation
        extrinsics_cw = np.eye(4, dtype=np.float32)
        extrinsics_cw[:3, :3] = R_cw
        extrinsics_cw[:3, 3] = cam_center[0]
        return jnp.array(extrinsics_cw, dtype=jnp.float32)

    def compute_projection_matrix(self, intr: CamIntrinsics, extr: CamExtrinsics):
        """
        Compute the projection matrix P = K * [R | t],
        where K is the camera intrinsics matrix.
        """
        K = intr.undist_cam_mtx.astype(np.float32)  # shape (3,3)
        RT = extr.transformation_matrix.astype(np.float32)  # shape (3,4)
        P = K @ RT  # Matrix multiplication yields (3,4)
        return jnp.array(P)

    def dmap2(self, dmap, name: str):
        depth_map_vis = dmap[2500:4500, 1200:2800].copy()
        plt.figure(figsize=(8, 6), dpi=400)
        plt.imshow(depth_map_vis, cmap="viridis")
        plt.colorbar(label="Depth")
        plt.clim(dmap[dmap != 0].min(), dmap.max())
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.savefig(f"_depth_map_to_center_{name}.png")

    @staticmethod
    @partial(
        jax.jit,
        static_argnums=(0, 1),
        device=jax.devices("cpu")[0],
    )
    def voxel_carving(voxel_shape, voxel_bounds, projection_matrices, silhouettes):
        """
        Voxel carving for multiple views implemented in JAX.

        Args:
            voxel_shape: Tuple (nx, ny, nz) defining the voxel grid resolution.
            voxel_bounds: Tuple ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in world coordinates.
            projection_matrices: JAX array of shape (M, 3, 4) for M camera projections.
            silhouettes: JAX array of shape (M, H, W) representing M binary masks.

        Returns:
            occupancy_grid: Boolean JAX array with shape voxel_shape (True = voxel survives).
            voxel_coords: JAX array of shape (N, 3) containing the world coordinates of each voxel center.
        """
        nx, ny, nz = voxel_shape
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = voxel_bounds

        # Create 1D arrays for the voxel centers.
        xs = jnp.linspace(xmin, xmax, nx)
        ys = jnp.linspace(ymin, ymax, ny)
        zs = jnp.linspace(zmin, zmax, nz)

        # Create a 3D grid (with 'ij' indexing so that first dim is x, etc.).
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        voxel_coords = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)  # (N, 3)
        N = voxel_coords.shape[0]

        # Convert voxel centers to homogeneous coordinates (N, 4)
        ones = jnp.ones((N, 1), dtype=voxel_coords.dtype)
        voxels_hom = jnp.concatenate([voxel_coords, ones], axis=1)  # (N, 4)

        # Define a function to process a single view.
        def process_view(proj_mat, silhouette, voxels_hom):
            # Project voxel centers: (N,4) x (4,3) -> (N, 3)
            proj = jnp.dot(voxels_hom, proj_mat.T)
            # Perform homogeneous division to get pixel coordinates.
            u = proj[:, 0] / proj[:, 2]
            v = proj[:, 1] / proj[:, 2]
            # Convert to integer pixel indices.
            u_int = jnp.floor(u).astype(jnp.int32)
            v_int = jnp.floor(v).astype(jnp.int32)
            # Get silhouette dimensions.
            H, W = silhouette.shape
            # Check if indices fall inside the image.
            inside = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
            # Compute flat indices for the silhouette.
            flat_indices = v_int * W + u_int
            silhouette_flat = silhouette.ravel()
            sil_values = jnp.where(inside, silhouette_flat[flat_indices], 0)
            # A voxel is valid for this view if it projects inside the silhouette (value > 0)
            # and its projected depth is positive.
            valid = (sil_values > 0) & (proj[:, 2] > 0)
            return valid  # (N,)

        # Use vmap to apply the processing over all views.
        # in_axes: projection_matrices and silhouettes are mapped over their first axis,
        # while voxels_hom is kept the same for all views.
        view_valid = jax.vmap(process_view, in_axes=(0, 0, None))(
            projection_matrices, silhouettes, voxels_hom
        )  # shape: (M, N)

        # For voxel carving across multiple views, we keep a voxel only if it is valid in all views.
        valid_all = jnp.all(view_valid, axis=0)  # shape: (N,)

        # Reshape the valid mask into the voxel grid.
        occupancy_grid = valid_all.reshape((nx, ny, nz))
        return occupancy_grid, voxel_coords

    @staticmethod
    @partial(jax.jit, device=jax.devices("cpu")[0])
    def fit_ear_center(valid_voxel_coords):

        # Use the z-coordinate as the parameter t (maize roughly vertical)
        t = valid_voxel_coords[:, 2]

        # Build the design matrix A with columns [t^2, t, 1]
        A = jnp.stack([t**2, t, jnp.ones_like(t)], axis=1)

        # Extract x and y coordinates from the valid_voxel_coords
        x_vals = valid_voxel_coords[:, 0]
        y_vals = valid_voxel_coords[:, 1]

        # Solve the least squares problems for x(t) and y(t)
        # x(t) = a*t^2 + b*t + c
        coeffs_x, residuals_x, rank_x, s_x = jnp.linalg.lstsq(A, x_vals, rcond=None)
        # y(t) = d*t^2 + e*t + f
        coeffs_y, residuals_y, rank_y, s_y = jnp.linalg.lstsq(A, y_vals, rcond=None)
        return jax.device_get(coeffs_x), jax.device_get(coeffs_y)

    def project_polynomial_to_image(
        self, coeffs_x, coeffs_y, projection_matrix, t_range, num_samples=100
    ):
        """
        Project the 3D polynomial curve onto a 2D image given a projection matrix.

        Parameters:
            coeffs_x (np.ndarray): Coefficients [a, b, c] for x(t).
            coeffs_y (np.ndarray): Coefficients [d, e, f] for y(t).
            projection_matrix (np.ndarray): A 3x4 projection matrix.
            t_range (tuple): (t_min, t_max) range for sampling the curve.
            num_samples (int): Number of sample points along t.

        Returns:
            u (np.ndarray): Array of u (horizontal) coordinates in the image.
            v (np.ndarray): Array of v (vertical) coordinates in the image.
        """
        # Sample t values along the curve
        t_values = np.linspace(t_range[0], t_range[1], num_samples)

        # Compute corresponding 3D points using the 3D polynomial
        x_values = coeffs_x[0] * t_values**2 + coeffs_x[1] * t_values + coeffs_x[2]
        y_values = coeffs_y[0] * t_values**2 + coeffs_y[1] * t_values + coeffs_y[2]
        z_values = t_values  # since t = z

        # Form homogeneous 3D points [x, y, z, 1] for each sample (shape: 4 x num_samples)
        points_3d = np.vstack([x_values, y_values, z_values, np.ones_like(t_values)])

        # Apply the projection matrix: resulting shape is (3, num_samples)
        proj_points = projection_matrix @ points_3d

        # Normalize homogeneous coordinates to get 2D image points
        u = proj_points[0, :] / proj_points[2, :]
        v = proj_points[1, :] / proj_points[2, :]
        return u, v

    @staticmethod
    @partial(jax.jit, device=jax.devices(GPU_DEVICE)[0])
    def project_xyz_on_images(
        candidate_points,
        poly_coeffs_x,
        poly_coeffs_y,
        camera_position,
        projection_matrix,
    ):
        """
        Projects candidate (x,y,z) points onto an image using a projection matrix,
        but only if the candidate point is closer to the camera than the corresponding
        point on the fitted polynomial curve at the same z-value.

        Parameters:
            candidate_points (jnp.ndarray): (N, 3) array of candidate (x,y,z) points.
            poly_coeffs_x (jnp.ndarray): (3,) coefficients [a, b, c] for x_poly(z) = a*z^2 + b*z + c.
            poly_coeffs_y (jnp.ndarray): (3,) coefficients [d, e, f] for y_poly(z) = d*z^2 + e*z + f.
            camera_position (jnp.ndarray): (3,) array for the camera position (x,y,z).
            projection_matrix (jnp.ndarray): (3,4) projection matrix.

        Returns:
            proj_points_masked (jnp.ndarray): (N, 2) array of 2D projected points.
                                            Invalid points (those that fail the distance test)
                                            are set to -1.
            valid_mask (jnp.ndarray): (N,) boolean mask where True indicates that the candidate
                                    point passed the distance test.
        """
        # Unpack candidate points
        x = candidate_points[:, 0]
        y = candidate_points[:, 1]
        z = candidate_points[:, 2]

        # Compute the corresponding point on the fitted polynomial at candidate z:
        poly_x = poly_coeffs_x[0] * z**2 + poly_coeffs_x[1] * z + poly_coeffs_x[2]
        poly_y = poly_coeffs_y[0] * z**2 + poly_coeffs_y[1] * z + poly_coeffs_y[2]

        # Compute squared Euclidean distances from the camera to candidate points.
        cam_x, cam_y, cam_z = camera_position
        d_candidate_sq = (x - cam_x) ** 2 + (y - cam_y) ** 2 + (z - cam_z) ** 2

        # Compute squared Euclidean distances from the camera to the polynomial points.
        d_poly_sq = (poly_x - cam_x) ** 2 + (poly_y - cam_y) ** 2 + (z - cam_z) ** 2

        # substract distances to qunatify occlusion
        delta_candidate_poly = jnp.sqrt(d_candidate_sq) - jnp.sqrt(d_poly_sq)

        # Create a boolean mask: True if candidate point is closer than the polynomial point.
        valid_mask = d_candidate_sq < d_poly_sq

        # Compute homogeneous coordinates for all candidate points
        ones = jnp.ones_like(z)
        points_hom = jnp.stack([x, y, z, ones], axis=1)  # Shape: (N,4)

        # Project points using the 3x4 projection matrix
        proj_hom = (projection_matrix @ points_hom.T).T  # Shape: (N,3)

        # Normalize homogeneous coordinates to get image coordinates (u,v)
        proj_points = proj_hom[:, :2] / proj_hom[:, 2:3]  # Shape: (N,2)

        # Option: For candidate points that are not valid, set their projection to -1.
        proj_points_masked = jnp.where(
            valid_mask[:, None], proj_points, -jnp.ones_like(proj_points)
        )
        return proj_points_masked, valid_mask, delta_candidate_poly

    @staticmethod
    @partial(jax.jit, static_argnums=(4), device=jax.devices(GPU_DEVICE)[0])
    def match_img_coords_to_voxel_coords(
        proj_matrix, valid_voxel_coords, img_coords, camera_center, max_match_dist
    ):
        """
        Computes a 2D distance map for a given silhouette where, for each white pixel,
        the output is the 3D distance (from the camera) to the closest voxel (from the
        carved voxel set) among those whose 2D projection is within 5 pixels.
        If no voxel is within that 5 pixel range, 0 is returned.

        Args:
            proj_matrix: JAX array of shape (3,4) representing the camera projection.
            valid_voxel_coords: JAX array of shape (N, 3) containing voxel centers in world coordinates.
            img_coords: JAX array of shape (P, 2) containing (row, col) indices of white pixels.
            camera_center: JAX array of shape (3,) with the camera center in world coordinates.

        Returns:
            distance_map: JAX array of shape (H, W) where for each white pixel the value is the
                        3D distance from the camera to the closest voxel (within 5 pixels on the image plane).
                        Non-white pixels remain 0.
        """
        # Number of valid voxels.
        M = valid_voxel_coords.shape[0]

        # Convert voxel centers to homogeneous coordinates.
        valid_voxel_coords_hom = jnp.concatenate(
            [valid_voxel_coords, jnp.ones((M, 1), dtype=valid_voxel_coords.dtype)],
            axis=1,
        )  # Shape (M, 4)

        # Project the valid voxels into the image.
        proj = jnp.dot(valid_voxel_coords_hom, proj_matrix.T)  # Shape (M, 3)
        # Perform homogeneous division.
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]
        voxel_proj = jnp.stack(
            [u, v], axis=-1
        )  # Shape (M, 2) where (u, v) are pixel coordinates

        # Compute the 3D (Euclidean) distance from the camera to each valid voxel.
        voxel_distances = jnp.linalg.norm(valid_voxel_coords - camera_center, axis=1)

        # Convert white_coords from (row, col) to (u, v) i.e. (col, row).
        white_pixels_uv = img_coords[:, ::-1]  # Shape (P, 2)

        # Define a function that, given a white pixel coordinate,
        # returns the 3D distance to the voxel that is both within 5 pixels on the image plane
        # and is the closest (in 3D) among those candidates.
        def min_distance_for_pixel(pixel_uv, max_match_dist):
            # Compute the Euclidean distance in the image plane between the pixel and all voxel projections.
            dists = jnp.linalg.norm(voxel_proj - pixel_uv, axis=1)
            # Create a mask: only consider voxels within 5 pixels.
            mask = dists <= max_match_dist
            any_candidate = jnp.any(mask)

            # If candidates exist, select the candidate with the minimum 3D distance.
            def if_true(_):
                candidate_voxel_distances = jnp.where(mask, voxel_distances, jnp.inf)
                return jnp.argmin(candidate_voxel_distances)

            # If no candidate is found, return 0.
            def if_false(_):
                return -1

            return jax.lax.cond(any_candidate, if_true, if_false, operand=None)

        min_pixel_distance_idcs = jax.vmap(min_distance_for_pixel, in_axes=(0, None))(
            white_pixels_uv, max_match_dist
        )
        return jax.device_get(min_pixel_distance_idcs)

    @staticmethod
    @partial(jax.jit, device=jax.devices(GPU_DEVICE)[0])
    def _process_batch_dmap_to_center(
        batch, valid_voxel_coords, voxel_proj, voxel_distances, coeffs_x, coeffs_y
    ):
        """
        Given a batch of white pixels (each a (u,v) coordinate), vectorize the
        min_distance_for_pixel function over the batch.

        """

        def min_distance_for_pixel(
            pixel_uv,
            valid_voxel_coords,
            voxel_proj,
            voxel_distances,
            coeffs_x,
            coeffs_y,
        ):
            """
            For a given white pixel (in 2D image space) compute:
            - the 2D distances from the pixel to every voxel projection,
            - select only those within 10 pixels,
            - and, if any exist, return the minimum 3D distance (from camera) among them.
            Otherwise, return 0.
            """
            # Compute 2D distances from the pixel to every voxel projection.
            dists = jnp.linalg.norm(voxel_proj - pixel_uv, axis=1)
            # Only consider voxels within 10 pixels.
            mask = dists <= 5.0
            any_candidate = jnp.any(mask)

            def fitted_x(t_val):
                return coeffs_x[0] * t_val**2 + coeffs_x[1] * t_val + coeffs_x[2]

            def fitted_y(t_val):
                return coeffs_y[0] * t_val**2 + coeffs_y[1] * t_val + coeffs_y[2]

            def fitted_curve(t_val):
                """Returns the 3D point [x(t), y(t), t] on the fitted curve."""
                return jnp.array([fitted_x(t_val), fitted_y(t_val), t_val])

            def if_true(_):
                # Replace voxels outside the range with infinity
                candidate_voxel_distances = jnp.where(mask, voxel_distances, jnp.inf)
                voi_indx = jnp.argmin(candidate_voxel_distances)
                # if candidate_voxel_distances[voi_indx] == jnp.inf:
                # return 0.0
                voi = valid_voxel_coords[voi_indx]
                return jnp.linalg.norm(voi - fitted_curve(voi[2]))

            def if_false(_):
                return 0.0

            return jax.lax.cond(any_candidate, if_true, if_false, operand=None)

        return jax.vmap(
            lambda pixel_uv: min_distance_for_pixel(
                pixel_uv,
                valid_voxel_coords,
                voxel_proj,
                voxel_distances,
                coeffs_x,
                coeffs_y,
            )
        )(batch)

    @staticmethod
    @partial(jax.jit, device=jax.devices(GPU_DEVICE)[0])
    def _process_batch_dmap_to_camera(batch, voxel_proj, voxel_distances):
        """
        Given a batch of white pixels (each a (u,v) coordinate), vectorize the
        min_distance_for_pixel function over the batch.

        """

        def min_distance_for_pixel(pixel_uv, voxel_proj, voxel_distances):
            """
            For a given white pixel (in 2D image space) compute:
            - the 2D distances from the pixel to every voxel projection,
            - select only those within 10 pixels,
            - and, if any exist, return the minimum 3D distance (from camera) among them.
            Otherwise, return 0.
            """
            # Compute 2D distances from the pixel to every voxel projection.
            dists = jnp.linalg.norm(voxel_proj - pixel_uv, axis=1)
            # Only consider voxels within 10 pixels.
            mask = dists <= 5.0
            any_candidate = jnp.any(mask)

            def if_true(_):
                # Replace voxels outside the range with infinity
                candidate_voxel_distances = jnp.where(mask, voxel_distances, jnp.inf)
                return jnp.min(candidate_voxel_distances)

            def if_false(_):
                return 0.0

            return jax.lax.cond(any_candidate, if_true, if_false, operand=None)

        return jax.vmap(
            lambda pixel_uv: min_distance_for_pixel(
                pixel_uv, voxel_proj, voxel_distances
            )
        )(batch)

    def projection_based_distance_map(
        self,
        proj_matrix,
        valid_voxel_coords,
        white_coords,
        silhouette_shape,
        camera_center,
        coeffs: None,
        batch_size=1024,
    ):
        """
        Computes a 2D distance map for a given silhouette where, for each white pixel,
        the output is the 3D distance (from the camera) to the closest voxel (from the carved set)
        among those whose 2D projection is within 10 pixels. If no voxel is within 10 pixels,
        a distance of 0 is returned.

        Args:
            proj_matrix: JAX array of shape (3,4) representing the camera projection.
            valid_voxel_coords: JAX array of shape (N, 3) containing voxel centers in world coordinates.
            white_coords: JAX array of shape (P, 2) containing (row, col) indices of white pixels.
            silhouette_shape: Tuple (H, W) representing the output image dimensions.
            camera_center: JAX array of shape (3,) with the camera center in world coordinates.
            batch_size: Integer batch size for processing white pixels.

        Returns:
            distance_map: JAX array of shape (H, W) where each white pixel is assigned the
                        computed 3D distance, and non-white pixels are 0.
        """
        if coeffs is not None:
            coeffs_x, coeffs_y = coeffs
        # --- Projection and 3D Distance Precomputation ---
        M = valid_voxel_coords.shape[0]
        # Convert voxel centers to homogeneous coordinates.
        valid_voxel_coords_hom = jnp.concatenate(
            [valid_voxel_coords, jnp.ones((M, 1), dtype=valid_voxel_coords.dtype)],
            axis=1,
        )  # Shape (M, 4)

        # Project the valid voxels into the image.
        proj = jnp.dot(valid_voxel_coords_hom, proj_matrix.T)  # Shape (M, 3)
        # Perform homogeneous division.
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]
        voxel_proj = jnp.stack([u, v], axis=-1)  # Shape (M, 2)

        # Compute the 3D Euclidean distance from the camera to each valid voxel.
        voxel_distances = jnp.linalg.norm(valid_voxel_coords - camera_center, axis=1)

        # --- Convert White Coordinates ---
        # white_coords are provided as (row, col); convert to (u, v) i.e. (col, row)
        white_pixels_uv = white_coords[:, ::-1]  # Shape (P, 2)

        # --- Batch Processing ---
        # Use standard Python slicing over white_pixels_uv (its shape is assumed to be statically known)
        P = white_pixels_uv.shape[
            0
        ]  # Should be a concrete integer if white_coords is static.
        batch_results = []
        for start in range(0, P, batch_size):
            end = start + batch_size
            # Use Python slicing; these indices are concrete integers.
            batch = white_pixels_uv[start:end, :]
            # Process this batch with our jitted function.
            if coeffs is not None:
                batch_result = self._process_batch_dmap_to_center(
                    batch,
                    valid_voxel_coords,
                    voxel_proj,
                    voxel_distances,
                    coeffs_x,
                    coeffs_y,
                )
            else:
                batch_result = self._process_batch_dmap_to_camera(
                    batch, voxel_proj, voxel_distances
                )
            batch_results.append(batch_result)
        # Concatenate all batch results (a (P,) array).
        pixel_distances = jnp.concatenate(batch_results, axis=0)

        # --- Build the Full 2D Distance Map ---
        H, W = silhouette_shape
        distance_map = jnp.zeros((H, W), dtype=valid_voxel_coords.dtype)
        # white_coords provides the (row, col) positions for white pixels.
        distance_map = distance_map.at[white_coords[:, 0], white_coords[:, 1]].set(
            pixel_distances
        )

        # Force computation and transfer result from device.
        return jax.device_get(distance_map)

    @staticmethod
    @partial(jax.jit, static_argnums=(2), device=jax.devices(GPU_DEVICE)[0])
    def compute_distances_and_filter(
        new_coords: jnp.ndarray, prev_coords: jnp.ndarray, thresh: float = 1.5
    ):
        """
        Compute pairwise Euclidean distances between points in array new_coords and array `prev_coord`,
        then filter points from `new_coords` whose distances to all points in `prev_coord` are above a given threshold.

        Parameters:
            new_coords (jnp.ndarray): An array of shape (N, 3) with xyz coordinates.
            prev_coord (jnp.ndarray): An array of shape (M, 3) with xyz coordinates.
            thresh (float): The distance threshold in mm.

        Returns:
            distances (jnp.ndarray): A (N, M) array of pairwise Euclidean distances.
            filter_mask (jnp.ndarray): A boolean mask of shape (N,) where True indicates that the point is far enough from all points in `prev_coords`.
        """

        # Compute the pairwise differences between points in a and b
        diff = new_coords[:, None, :] - prev_coords[None, :, :]  # shape (N, M, 3)
        # Compute Euclidean distances along the last dimension
        distances = jnp.linalg.norm(diff, axis=-1)  # shape (N, M)

        # Create a boolean mask: True for each point in `a` whose distances to all points in `b` exceed thresh
        filter_mask = jnp.all(distances > thresh, axis=1)  # shape (N,)

        return distances, filter_mask

    @staticmethod
    @partial(jax.jit, device=jax.devices(GPU_DEVICE)[0])
    def compute_px_per_mm2_scales(xyz, pm):
        """
        Computes the local scale factor (pixels per mm²) for each 3D world point in xyz,
        without assuming the points lie exactly on a common plane.

        The function computes the full 3D Jacobian of the projection function
        (from R^3 to R^2) and then uses singular value decomposition to determine the
        effective area scaling. The product of the two nonzero singular values is taken as
        the local area scale (i.e. pixels per mm²) at that point.

        Parameters:
            xyz: jnp.array of shape (N, 3)
                Array of 3D world coordinates.
            pm: jnp.array of shape (3, 4)
                Camera projection matrix.

        Returns:
            scales: jnp.array of shape (N,)
                Local area scale factors for each point.
        """

        def project_fn(world_point):
            # Append 1 to form the homogeneous coordinate.
            point = jnp.concatenate([world_point, jnp.array([1.0])])
            proj = pm @ point  # shape (3,)
            u = proj[0] / proj[2]
            v = proj[1] / proj[2]
            return jnp.array([u, v])

        def scale_for_point(point):
            # Compute the full 2x3 Jacobian of the projection function.
            J = jax.jacobian(project_fn)(point)  # J has shape (2, 3)
            # Compute the singular values of J.
            # For a well-behaved projection, J is rank 2.
            S = jnp.linalg.svd(J, compute_uv=False)
            # The product of the two nonzero singular values gives the local area scaling.
            return S[0] * S[1]

        scales = jax.vmap(scale_for_point)(xyz)
        return scales
