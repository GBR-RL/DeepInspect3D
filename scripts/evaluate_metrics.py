import open3d as o3d
import numpy as np
import argparse
import logging
import time
import json
from pathlib import Path


def setup_logging(verbose: bool):
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)


def safe_read_pcd(path: Path) -> o3d.geometry.PointCloud:
    """Load a point cloud, ensure it exists, is non-empty, and remove non-finite points."""
    if not path.is_file():
        logging.error("File not found: %s", path)
        raise FileNotFoundError(f"No such file: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        logging.error("Empty point cloud: %s", path)
        raise ValueError(f"Empty point cloud: {path}")
    pcd, ind_nf = pcd.remove_non_finite_points()
    if len(ind_nf) > 0:
        logging.info("Removed %d non-finite points", len(ind_nf))
    return pcd


def compute_mean_surface_deviation(
    test: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud
) -> tuple[float, float]:
    """Compute mean and max per-point deviation from test to reference."""
    if test.is_empty():
        raise ValueError("Test point cloud is empty")
    if reference.is_empty():
        raise ValueError("Reference point cloud is empty")
    distances = np.asarray(test.compute_point_cloud_distance(reference))
    mean_dev = distances.mean() if distances.size > 0 else 0.0
    max_dev = distances.max() if distances.size > 0 else 0.0
    return mean_dev, max_dev


def compute_chamfer_distance(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud
) -> tuple[float, float, float]:
    """Compute symmetric Chamfer distance between two clouds."""
    if pcd1.is_empty() or pcd2.is_empty():
        raise ValueError("Point clouds for Chamfer computation must be non-empty")
    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    cd1 = d1.mean() if d1.size > 0 else 0.0
    cd2 = d2.mean() if d2.size > 0 else 0.0
    cd_avg = (cd1 + cd2) / 2.0
    return cd1, cd2, cd_avg


def evaluate_precision_recall(
    predicted: o3d.geometry.PointCloud,
    ground_truth: o3d.geometry.PointCloud,
    threshold: float
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1-score of predicted defects."""
    if threshold <= 0:
        raise ValueError("Threshold must be > 0")
    pred_count = len(predicted.points)
    gt_count = len(ground_truth.points)
    if pred_count == 0 or gt_count == 0:
        logging.warning(
            "Empty mask(s): predicted=%d, ground_truth=%d; precision/recall set to 0", 
            pred_count, gt_count
        )
        return 0.0, 0.0, 0.0
    d_pred_to_gt = np.asarray(predicted.compute_point_cloud_distance(ground_truth))
    tp = np.count_nonzero(d_pred_to_gt < threshold)
    precision = tp / pred_count
    d_gt_to_pred = np.asarray(ground_truth.compute_point_cloud_distance(predicted))
    tp2 = np.count_nonzero(d_gt_to_pred < threshold)
    recall = tp2 / gt_count
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate metrics between test and reference point clouds"
    )
    parser.add_argument(
        '--test', type=Path, required=True,
        help='Aligned test PLY file'
    )
    parser.add_argument(
        '--reference', type=Path, required=True,
        help='Reference PLY file'
    )
    parser.add_argument(
        '--pred_mask', type=Path,
        help='Predicted defect mask PLY file'
    )
    parser.add_argument(
        '--gt_mask', type=Path,
        help='Ground truth defect mask PLY file'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.001,
        help='Threshold for PR evaluation (in meters). Must be > 0.'
    )
    parser.add_argument(
        '--output_metrics', type=Path,
        help='Optional path to save metrics JSON'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    total_start = time.perf_counter()

    # Load and validate point clouds
    test_pcd = safe_read_pcd(args.test)
    ref_pcd = safe_read_pcd(args.reference)

    # Mean surface deviation
    t0 = time.perf_counter()
    mean_dev, max_dev = compute_mean_surface_deviation(test_pcd, ref_pcd)
    logging.info(
        "Surface Deviation → mean: %.6f, max: %.6f [%.3fs]",
        mean_dev, max_dev, time.perf_counter() - t0
    )

    # Chamfer distance
    t1 = time.perf_counter()
    cd1, cd2, cd_avg = compute_chamfer_distance(test_pcd, ref_pcd)
    logging.info(
        "Chamfer → test2ref: %.6f, ref2test: %.6f, avg: %.6f [%.3fs]",
        cd1, cd2, cd_avg, time.perf_counter() - t1
    )

    metrics = {
        "mean_surface_deviation": mean_dev,
        "max_surface_deviation": max_dev,
        "chamfer_test_to_ref": cd1,
        "chamfer_ref_to_test": cd2,
        "chamfer_avg": cd_avg
    }

    # Precision/Recall if masks provided
    if args.pred_mask and args.gt_mask:
        pred_pcd = safe_read_pcd(args.pred_mask)
        gt_pcd = safe_read_pcd(args.gt_mask)
        t2 = time.perf_counter()
        precision, recall, f1 = evaluate_precision_recall(
            pred_pcd, gt_pcd, args.threshold
        )
        logging.info(
            "Precision/Recall → precision: %.6f, recall: %.6f, f1: %.6f [%.3fs]",
            precision, recall, f1, time.perf_counter() - t2
        )
        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
    else:
        if args.pred_mask or args.gt_mask:
            logging.warning(
                "Both --pred_mask and --gt_mask must be provided for PR evaluation; skipping"
            )

    total_time = time.perf_counter() - total_start
    metrics["total_time_s"] = total_time
    logging.info("Total evaluation time: %.3f s", total_time)

    # Save metrics if requested
    if args.output_metrics:
        out_dir = args.output_metrics.parent
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info("Saved metrics JSON to %s", args.output_metrics)

if __name__ == '__main__':
    main()
