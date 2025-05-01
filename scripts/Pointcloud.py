import open3d as o3d
import logging
import numpy as np
import os
import json
import argparse

def setup_logging(verbose: bool):
    """Configure root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)

def load_intrinsics(json_path: str) -> dict:
    """Load & validate the intrinsics JSON produced by render.py."""
    if not os.path.isfile(json_path):
        logging.error("Intrinsics file not found: %s", json_path)
        raise FileNotFoundError(f"{json_path} not found")

    with open(json_path, 'r') as f:
        meta = json.load(f)

    # Top‐level keys
    for k in ("image_width", "image_height", "camera_matrix", "depth_scale"):
        if k not in meta:
            logging.error("Missing key '%s' in intrinsics JSON", k)
            raise KeyError(f"'{k}' missing in {json_path}")

    # Minimal nested check via try/except
    try:
        cm = meta["camera_matrix"]
        fx = cm["fx"]; fy = cm["fy"]
        cx = cm["cx"]; cy = cm["cy"]
    except KeyError as e:
        logging.error("Missing camera_matrix.%s in %s", e.args[0], json_path)
        raise KeyError(f"camera_matrix.{e.args[0]} missing in {json_path}")

    return meta

def convert_depth_to_pointcloud(
    depth_path: str,
    intrinsics_json: str,
    output_path: str,
    depth_trunc: float = None
) -> o3d.geometry.PointCloud:
    logging.info("Converting depth→PCD: %s", depth_path)
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(depth_path)

    depth_raw = o3d.io.read_image(depth_path)
    logging.debug("Loaded depth (%d×%d)", depth_raw.width, depth_raw.height)

    meta = load_intrinsics(intrinsics_json)
    cam = o3d.camera.PinholeCameraIntrinsic(
        meta['image_width'], meta['image_height'],
        meta['camera_matrix']['fx'], meta['camera_matrix']['fy'],
        meta['camera_matrix']['cx'], meta['camera_matrix']['cy']
    )

    # log valid pixels
    depth_np = np.asarray(depth_raw)
    valid = np.count_nonzero(depth_np)
    logging.debug("Valid depth pixels: %d/%d", valid, depth_np.size)

    kwargs = {
        'depth_scale': meta['depth_scale'],
        'project_valid_depth_only': True
    }
    if depth_trunc is not None:
        kwargs['depth_trunc'] = depth_trunc

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_raw, cam, **kwargs
    )
    logging.debug("Point cloud has %d points", len(pcd.points))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    o3d.io.write_point_cloud(output_path, pcd)
    logging.info("Wrote PLY → %s", output_path)
    return pcd


def convert_with_color(
    depth_path: str,
    color_path: str,
    intrinsics_json: str,
    output_path: str
) -> o3d.geometry.PointCloud:
    logging.info("Converting RGBD→PCD: depth=%s, color=%s", depth_path, color_path)
    for p in (depth_path, color_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    depth_raw = o3d.io.read_image(depth_path)
    color_raw = o3d.io.read_image(color_path)
    logging.debug("Depth %dx%d, Color %dx%d",
                  depth_raw.width, depth_raw.height,
                  color_raw.width, color_raw.height)

    if (depth_raw.width, depth_raw.height)!=(color_raw.width, color_raw.height):
        raise ValueError(
            f"Depth ({depth_raw.width}×{depth_raw.height}) "
            f"vs Color ({color_raw.width}×{color_raw.height}) mismatch"
        )

    meta = load_intrinsics(intrinsics_json)
    cam = o3d.camera.PinholeCameraIntrinsic(
        meta['image_width'], meta['image_height'],
        meta['camera_matrix']['fx'], meta['camera_matrix']['fy'],
        meta['camera_matrix']['cx'], meta['camera_matrix']['cy']
    )

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        depth_scale=meta['depth_scale'],
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)
    logging.info("RGBD→PCD has %d points", len(pcd.points))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert depth maps to .ply point clouds (with optional color)"
    )
    parser.add_argument("--depth",      required=True, help="16-bit PNG depth map")
    parser.add_argument("--intrinsics", required=True, help="Intrinsics JSON file")
    parser.add_argument("--output",     required=True, help="Output PLY path")
    parser.add_argument("--color",      help="Optional color image for RGBD conversion")
    parser.add_argument("--verbose",    action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logging.info("Point cloud conversion script started")

    try:
        if args.color:
            convert_with_color(args.depth, args.color, args.intrinsics, args.output)
        else:
            convert_depth_to_pointcloud(args.depth, args.intrinsics, args.output)
    except Exception as e:
        logging.exception("Conversion failed: %s", e)
        exit(1)

    logging.info("Script finished successfully")
