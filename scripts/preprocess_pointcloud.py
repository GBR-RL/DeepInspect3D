import open3d as o3d
import argparse
import os
import logging
import time
import yaml

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)

def safe_read_pcd(path: str) -> o3d.geometry.PointCloud:
    if not os.path.isfile(path):
        logging.error("Input file not found: %s", path)
        raise FileNotFoundError(f"No such file: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        logging.error("Loaded point cloud is empty: %s", path)
        raise ValueError(f"Empty point cloud: {path}")
    return pcd

def load_preprocess_cfg(cfg_path: str) -> dict:
    """Load and validate preprocessing parameters from a YAML config."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    pre = cfg.get("preprocessing", {}) or {}
    required = [
        "voxel_size",
        "stat_nb_neighbors",
        "stat_std_ratio",
        "radius_search",
        "normal_max_nn",
        "use_radius_outlier",
        "radius_nb_points"
    ]
    missing = [k for k in required if k not in pre]
    if missing:
        raise KeyError(f"Missing keys in [preprocessing] config: {missing}")
    return pre

def preprocess_pointcloud(
    input_path: str,
    output_path: str,
    voxel_size: float,
    stat_nb_neighbors: int,
    stat_std_ratio: float,
    radius_search: float,
    normal_max_nn: int,
    use_radius_outlier: bool,
    radius_nb_points: int
) -> None:
    """
    Apply voxel downsampling, outlier removal, normal estimation,
    and save cleaned point cloud to disk.
    """
    t_start = time.perf_counter()

    # 0) Load and remove non-finite points
    pcd = safe_read_pcd(input_path)
    pcd, ind_nf = pcd.remove_non_finite_points()
    logging.info("Removed %d non-finite points", len(ind_nf))

    # 1) Voxel downsampling
    t0 = time.perf_counter()
    pcd = pcd.voxel_down_sample(voxel_size)
    logging.info(
        "Downsampled: %d points (voxel_size=%.4f) in %.3f s",
        len(pcd.points), voxel_size, time.perf_counter() - t0
    )

    # 2) Statistical outlier removal
    t1 = time.perf_counter()
    pcd, ind_st = pcd.remove_statistical_outlier(
        nb_neighbors=stat_nb_neighbors,
        std_ratio=stat_std_ratio
    )
    logging.info(
        "Statistical filter: %d points kept (k=%d, std_ratio=%.2f) in %.3f s",
        len(pcd.points), stat_nb_neighbors, stat_std_ratio, time.perf_counter() - t1
    )

    # 3) Optional radius-based outlier removal
    if use_radius_outlier:
        t1b = time.perf_counter()
        pcd, ind_rb = pcd.remove_radius_outlier(
            nb_points=radius_nb_points,
            radius=radius_search  # reuse for normals too
        )
        logging.info(
            "Radius filter: %d points kept (nb_points=%d, radius=%.4f) in %.3f s",
            len(pcd.points), radius_nb_points, radius_search, time.perf_counter() - t1b
        )

    # 4) Estimate normals using hybrid search
    t2 = time.perf_counter()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_search,
            max_nn=normal_max_nn
        )
    )
    logging.info(
        "Normals estimated (radius=%.4f, max_nn=%d) in %.3f s",
        radius_search, normal_max_nn, time.perf_counter() - t2
    )

    # 5) Save output, guarding empty dir
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(
        output_path,
        pcd,
        write_ascii=False,
        compressed=False
    )
    logging.info(
        "Saved preprocessed cloud: %s (total time %.3f s)",
        output_path,
        time.perf_counter() - t_start
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess PLY point cloud: downsample, denoise, normals.'
    )
    parser.add_argument('--input', required=True, help='Input .ply file path')
    parser.add_argument('--output', required=True, help='Output .ply file path')
    parser.add_argument(
        '--config',
        help='YAML config file with a [preprocessing] section',
        default=None
    )
    parser.add_argument('--voxel_size',     type=float, default=0.005,
                        help='Voxel size for downsampling')
    parser.add_argument('--stat_nb_neighbors', type=int, default=20,
                        help='Statistical outlier K')
    parser.add_argument('--stat_std_ratio',    type=float, default=2.0,
                        help='Statistical std ratio')
    parser.add_argument('--use_radius_outlier', action='store_true',
                        help='Enable radius-based outlier removal')
    parser.add_argument('--radius_nb_points',  type=int, default=16,
                        help='Min neighbors for radius outlier')
    parser.add_argument('--radius_search',     type=float, default=0.02,
                        help='Radius for outlier and normal search')
    parser.add_argument('--normal_max_nn',     type=int, default=30,
                        help='Max neighbors for normal estimation')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    setup_logging(args.verbose)
    if args.config:
        try:
            cfg = load_preprocess_cfg(args.config)
            # Override CLI with YAML values
            args.voxel_size        = cfg['voxel_size']
            args.stat_nb_neighbors = cfg['stat_nb_neighbors']
            args.stat_std_ratio    = cfg['stat_std_ratio']
            args.radius_search     = cfg['radius_search']
            args.normal_max_nn     = cfg['normal_max_nn']
            args.use_radius_outlier= cfg['use_radius_outlier']
            args.radius_nb_points  = cfg['radius_nb_points']
        except Exception as e:
            logging.error("Failed loading config: %s", e)
            exit(1)

    try:
        preprocess_pointcloud(
            args.input,
            args.output,
            args.voxel_size,
            args.stat_nb_neighbors,
            args.stat_std_ratio,
            args.radius_search,
            args.normal_max_nn,
            args.use_radius_outlier,
            args.radius_nb_points
        )
    except Exception:
        logging.exception("Preprocessing failed")
        exit(1)