import open3d as o3d
import numpy as np
import argparse
import logging
import time
from pathlib import Path

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)

def safe_read_pcd(path: Path) -> o3d.geometry.PointCloud:
    """Load a point cloud, ensure it exists, is non-empty, and drop non-finite points."""
    if not path.is_file():
        logging.error("Input file not found: %s", path)
        raise FileNotFoundError(f"No such file: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        logging.error("Loaded point cloud is empty: %s", path)
        raise ValueError(f"Empty point cloud: {path}")
    # remove NaN / Inf
    pcd, ind_nf = pcd.remove_non_finite_points()
    if len(ind_nf) > 0:
        logging.info("Removed %d non-finite points", len(ind_nf))
    return pcd

def preprocess_for_registration(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_multiplier: float,
    feature_multiplier: float
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Downsample, estimate normals, and compute FPFH features."""
    # 1) Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 2) Normals
    radius_normal = voxel_size * normal_multiplier
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30
        )
    )
    logging.debug("Normals estimated on downsampled cloud (radius=%.3f)", radius_normal)

    # 3) FPFH feature extraction
    radius_feature = voxel_size * feature_multiplier
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100
        )
    )
    logging.debug("FPFH computed (radius=%.3f)", radius_feature)
    return pcd_down, fpfh

def execute_global_registration(
    src_down, tgt_down, src_fpfh, tgt_fpfh,
    voxel_size: float,
    use_fgr: bool,
    ransac_iter: int,
    ransac_valid: int,
    edge_length: float
) -> o3d.pipelines.registration.RegistrationResult:
    """Perform either RANSAC or Fast Global Registration."""
    if use_fgr:
        logging.info("Running Fast Global Registration")
        return o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            src_down, tgt_down,
            src_fpfh, tgt_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=voxel_size * 1.5
            )
        )
    else:
        logging.info("Running RANSAC Global Registration")
        distance_threshold = voxel_size * 1.5
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down, tgt_down,
            src_fpfh, tgt_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(edge_length),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iter, ransac_valid)
        )

def refine_registration(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init_transformation: np.ndarray,
    voxel_size: float,
    use_point_to_plane: bool
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine with ICP, ensuring normals exist on both source and target."""
    # Estimate normals on full-resolution clouds if needed
    for name, cloud in (("source", src), ("target", tgt)):
        if not cloud.has_normals():
            radius = voxel_size * 2.0
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
            logging.debug("Estimated normals on %s cloud (radius=%.3f)", name, radius)

    distance_threshold = voxel_size * 0.4
    method = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
        if use_point_to_plane else
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    logging.info(
        "Running ICP (%s) with max_correspondence_distance=%.4f",
        "PointToPlane" if use_point_to_plane else "PointToPoint",
        distance_threshold
    )
    return o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=distance_threshold,
        init=init_transformation,
        estimation_method=method
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register two point clouds via global + ICP refinement"
    )
    parser.add_argument("--source", required=True, type=Path,
                        help="Path to source (to be aligned) PLY")
    parser.add_argument("--target", required=True, type=Path,
                        help="Path to target (reference) PLY")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to save aligned source PLY")
    parser.add_argument("--matrix", type=Path, default=None,
                        help="Optional path to save transformation matrix (.npy)")
    parser.add_argument("--voxel_size", type=float, default=0.005,
                        help="Downsampling voxel size")
    parser.add_argument("--normal_mult", type=float, default=2.0,
                        help="Multiplier for normal estimation radius (voxel_size * normal_mult)")
    parser.add_argument("--feature_mult", type=float, default=5.0,
                        help="Multiplier for FPFH feature radius (voxel_size * feature_mult)")
    parser.add_argument("--use-fgr", action="store_true",
                        help="Use Fast Global Registration instead of RANSAC")
    parser.add_argument("--ransac_iter", type=int, default=4_000_000,
                        help="Max RANSAC iterations")
    parser.add_argument("--ransac_valid", type=int, default=500,
                        help="Max RANSAC validation steps")
    parser.add_argument("--edge_length", type=float, default=0.9,
                        help="Edge length checker threshold")
    parser.add_argument("--use-point-to-plane", action="store_true",
                        help="Use point-to-plane ICP (requires normals)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    start = time.perf_counter()

    # Load & clean
    src_raw = safe_read_pcd(args.source)
    tgt_raw = safe_read_pcd(args.target)

    # Preprocess downsample + FPFH
    src_down, src_fpfh = preprocess_for_registration(
        src_raw, args.voxel_size, args.normal_mult, args.feature_mult
    )
    tgt_down, tgt_fpfh = preprocess_for_registration(
        tgt_raw, args.voxel_size, args.normal_mult, args.feature_mult
    )

    # Global alignment
    t0 = time.perf_counter()
    result_global = execute_global_registration(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        args.voxel_size, args.use_fgr,
        args.ransac_iter, args.ransac_valid, args.edge_length
    )
    logging.info(
        "Global registration: fitness=%.4f, inlier_rmse=%.4f [%.2fs]",
        result_global.fitness, result_global.inlier_rmse,
        time.perf_counter() - t0
    )

    # ICP refinement
    t1 = time.perf_counter()
    result_icp = refine_registration(
        src_raw, tgt_raw,
        result_global.transformation,
        args.voxel_size,
        args.use_point_to_plane
    )
    logging.info(
        "ICP refinement: fitness=%.4f, inlier_rmse=%.4f [%.2fs]",
        result_icp.fitness, result_icp.inlier_rmse,
        time.perf_counter() - t1
    )

    # Apply and save
    src_raw.transform(result_icp.transformation)
    out_dir = args.output.parent
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.output), src_raw)
    logging.info("Saved aligned cloud to %s [total %.2fs]", args.output, time.perf_counter() - start)

    # Save matrix if requested
    if args.matrix:
        mat_dir = args.matrix.parent
        if mat_dir:
            mat_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(args.matrix), result_icp.transformation)
        logging.info("Saved transformation matrix to %s", args.matrix)
