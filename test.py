import argparse
import os
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import random
import json
from multiprocessing import Pool, Queue, Process, Manager
import queue
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator
import warnings

warnings.filterwarnings('ignore')


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

g_label2color = {
    0: [0, 255, 0],
    1: [255, 0, 0],
}

classes = ['ground', 'non-ground']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_label_to_cat = {i: cat for i, cat in enumerate(class2label.keys())}


def parse_args():
    parser = argparse.ArgumentParser('TSRNet V3 Testing')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=23)
    parser.add_argument('--num_point', type=int, default=8192)
    parser.add_argument('--num_votes', type=int, default=5)
    parser.add_argument('--block_size', type=float, default=32.0)
    parser.add_argument('--stride', type=float, default=16.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic_sampling', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--prefetch_batches', type=int, default=10)

    parser.add_argument('--log_dir', type=str, default='TSRNet')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pth')
    parser.add_argument('--kp_radius_scale', type=float, default=6.2)

    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)

    parser.add_argument('--test_file', type=str,
                        default='data/OpenGF_test/AreaT1_change_label_pianyi.npy')
    parser.add_argument('--test_all', action='store_true')

    parser.add_argument('--adaptive_features', action='store_true', default=True)
    parser.add_argument('--density_radius_min', type=float, default=0.8)
    parser.add_argument('--density_radius_max', type=float, default=3.5)
    parser.add_argument('--normal_k_extreme_sparse', type=int, default=5)
    parser.add_argument('--normal_k_sparse', type=int, default=10)
    parser.add_argument('--normal_k_normal', type=int, default=20)
    parser.add_argument('--normal_k_dense', type=int, default=28)
    parser.add_argument('--normal_k_extreme_dense', type=int, default=36)
    parser.add_argument('--normal_k', type=int, default=20)
    parser.add_argument('--density_radius', type=float, default=1.0)

    parser.add_argument('--analyze_scene', action='store_true', default=True)
    parser.add_argument('--monitor_uncertainty', action='store_true', default=True)

    parser.add_argument('--visual', action='store_true', default=True)
    parser.add_argument('--save_detailed_results', action='store_true', default=True)
    parser.add_argument('--save_predictions', action='store_true', default=True)

    parser.add_argument('--dtm_resolution', type=float, default=0.6)
    parser.add_argument('--skip_rmse', action='store_true')
    parser.add_argument('--compute_dger_rmse', action='store_true', default=True)

    return parser.parse_args()


def compute_adaptive_k(median_dist, args):
    if median_dist > 3.0:
        return args.normal_k_extreme_sparse, "Extreme Sparse"
    elif median_dist > 1.5:
        return args.normal_k_sparse, "Sparse"
    elif median_dist > 0.8:
        return args.normal_k_normal, "Normal"
    elif median_dist > 0.3:
        return args.normal_k_dense, "Dense"
    else:
        return args.normal_k_extreme_dense, "Extreme Dense"


def compute_full_normals_and_curvature(points, normal_k=20):
    n_points = points.shape[0]
    normals = np.zeros((n_points, 3), dtype=np.float32)
    curvatures = np.zeros(n_points, dtype=np.float32)
    k = min(normal_k, n_points)
    try:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=1).fit(points)
        distances, indices = nbrs.kneighbors(points)
        for i in range(n_points):
            try:
                neighbors = points[indices[i]]
                centroid = neighbors.mean(axis=0)
                centered = neighbors - centroid
                dists = np.linalg.norm(centered, axis=1)
                median_d = np.median(dists)
                weights = np.exp(-((dists - median_d) ** 2) / (2 * (median_d ** 2 + 1e-8)))
                weights = weights / (weights.sum() + 1e-8)
                weighted_centered = centered * weights[:, np.newaxis]
                cov = np.dot(weighted_centered.T, centered) / k
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]
                if normal[2] < 0:
                    normal = -normal
                normals[i] = normal
                eigenvalues = np.sort(eigenvalues)
                curvatures[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-8)
            except:
                normals[i] = np.array([0, 0, 1], dtype=np.float32)
                curvatures[i] = 0.0
    except:
        normals[:, 2] = 1.0
        curvatures[:] = 0.0
    return normals, curvatures


def compute_features_for_block(block_info, args):
    selected_points, original_indices, coord_max, num_point = block_info
    model_input = np.zeros((num_point, 11), dtype=np.float32)

    block_mean = np.mean(selected_points, axis=0)
    centered_xyz = selected_points - block_mean
    model_input[:, 0:3] = centered_xyz

    block_z_min = float(centered_xyz[:, 2].min())
    block_z_max = float(centered_xyz[:, 2].max())
    block_z_center = float(block_mean[2])

    model_input[:, 3] = selected_points[:, 0] / (coord_max[0] + 1e-8)
    model_input[:, 4] = selected_points[:, 1] / (coord_max[1] + 1e-8)
    model_input[:, 5] = selected_points[:, 2] / (coord_max[2] + 1e-8)
    try:
        tree = cKDTree(selected_points)
        if args.adaptive_features:
            sample_size = min(500, len(selected_points))
            sample_idx = np.random.choice(len(selected_points), sample_size, replace=False)
            distances, _ = tree.query(selected_points[sample_idx], k=20)
            median_dist = np.median(distances[:, -1])
            adaptive_radius = np.clip(
                median_dist * 3, args.density_radius_min, args.density_radius_max
            )
            adaptive_k, _ = compute_adaptive_k(median_dist, args)
            normals, curvatures = compute_full_normals_and_curvature(selected_points, adaptive_k)
            neighbor_counts = tree.query_ball_point(
                selected_points, r=adaptive_radius, return_length=True
            )
            density = np.array(neighbor_counts, dtype=np.float32)
            max_density = np.percentile(density, 95)
            if max_density > 0:
                density = np.clip(density / max_density, 0, 1.0)
            else:
                density = np.clip(density / 100.0, 0, 1.0)
        else:
            density = tree.query_ball_point(
                selected_points, r=args.density_radius, return_length=True
            ).astype(np.float32)
            density = np.clip(density / 100.0, 0, 1.0)
            normals, curvatures = compute_full_normals_and_curvature(
                selected_points, args.normal_k
            )
    except Exception:
        density = np.full(num_point, 0.5, dtype=np.float32)
        normals = np.tile([0, 0, 1], (num_point, 1)).astype(np.float32)
        curvatures = np.full(num_point, 0.1, dtype=np.float32)

    model_input[:, 6] = density
    model_input[:, 7:10] = normals
    model_input[:, 10] = curvatures

    return model_input, original_indices, block_z_min, block_z_max, block_z_center


def data_producer(block_queue, batch_queue, points, coord_min, coord_max, args, vote_idx):
    try:
        vote_seed = args.seed + vote_idx * 1000
        np.random.seed(vote_seed)
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - args.block_size) / args.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - args.block_size) / args.stride) + 1)
        for index_y in range(grid_y):
            for index_x in range(grid_x):
                s_x = coord_min[0] + index_x * args.stride
                e_x = min(s_x + args.block_size, coord_max[0])
                s_x = e_x - args.block_size
                s_y = coord_min[1] + index_y * args.stride
                e_y = min(s_y + args.block_size, coord_max[1])
                s_y = e_y - args.block_size
                point_idxs_in_block = np.where(
                    (points[:, 0] >= s_x) & (points[:, 0] <= e_x) &
                    (points[:, 1] >= s_y) & (points[:, 1] <= e_y)
                )[0]
                if point_idxs_in_block.size < 100:
                    continue
                if args.deterministic_sampling:
                    block_seed = vote_seed + index_x * 1000 + index_y
                    np.random.seed(block_seed)
                if len(point_idxs_in_block) >= args.num_point:
                    sampled_idxs_local = np.random.choice(
                        len(point_idxs_in_block), args.num_point, replace=False
                    )
                else:
                    sampled_idxs_local = np.random.choice(
                        len(point_idxs_in_block), args.num_point, replace=True
                    )
                original_indices = point_idxs_in_block[sampled_idxs_local]
                selected_points = points[original_indices, :]
                block_info = (selected_points, original_indices, coord_max, args.num_point)
                block_queue.put(block_info)
        for _ in range(args.num_workers):
            block_queue.put(None)
    except Exception as e:
        print(f"Producer error: {e}")


def feature_worker(block_queue, batch_queue, batch_size, args):
    try:
        batch_features = []
        batch_indices = []
        batch_z_mins = []
        batch_z_maxs = []
        batch_z_centers = []
        while True:
            block_info = block_queue.get()
            if block_info is None:
                break
            features, indices, z_min, z_max, z_center = compute_features_for_block(block_info, args)
            batch_features.append(features)
            batch_indices.append(indices)
            batch_z_mins.append(z_min)
            batch_z_maxs.append(z_max)
            batch_z_centers.append(z_center)
            if len(batch_features) >= batch_size:
                batch_queue.put((
                    np.stack(batch_features),
                    np.stack(batch_indices),
                    np.array(batch_z_mins, dtype=np.float32),
                    np.array(batch_z_maxs, dtype=np.float32),
                    np.array(batch_z_centers, dtype=np.float32)
                ))
                batch_features = []
                batch_indices = []
                batch_z_mins = []
                batch_z_maxs = []
                batch_z_centers = []
        if batch_features:
            batch_queue.put((
                np.stack(batch_features),
                np.stack(batch_indices),
                np.array(batch_z_mins, dtype=np.float32),
                np.array(batch_z_maxs, dtype=np.float32),
                np.array(batch_z_centers, dtype=np.float32)
            ))
    except Exception as e:
        print(f"Worker error: {e}")


def add_vote(vote_label_pool, point_indices_batch, pred_labels_batch):
    for i in range(point_indices_batch.shape[0]):
        point_indices = point_indices_batch[i]
        pred_labels = pred_labels_batch[i]
        for j in range(len(point_indices)):
            vote_label_pool[int(point_indices[j]), int(pred_labels[j])] += 1
    return vote_label_pool


def calculate_f1_scores(total_correct_class, total_seen_class, total_pred_class,
                        num_classes, class_names, eps=1e-6):
    precisions, recalls, f1_scores = [], [], []
    for l in range(num_classes):
        precision = total_correct_class[l] / (total_pred_class[l] + eps)
        recall = total_correct_class[l] / (total_seen_class[l] + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)
    total_tp = sum(total_correct_class)
    total_pred = sum(total_pred_class)
    total_true = sum(total_seen_class)
    micro_precision = total_tp / (total_pred + eps)
    micro_recall = total_tp / (total_true + eps)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + eps)
    return {
        'class_precision': precisions, 'class_recall': recalls, 'class_f1': f1_scores,
        'macro_precision': np.mean(precisions), 'macro_recall': np.mean(recalls),
        'macro_f1': np.mean(f1_scores),
        'micro_precision': micro_precision, 'micro_recall': micro_recall, 'micro_f1': micro_f1
    }


def calculate_rmse_dtm(points, true_labels, pred_labels, resolution=1.0):
    true_ground_mask = (true_labels == 0)
    pred_ground_mask = (pred_labels == 0)
    true_ground_points = points[true_ground_mask]
    pred_ground_points = points[pred_ground_mask]
    if len(true_ground_points) == 0 or len(pred_ground_points) == 0:
        return np.nan, 0.0
    x_min = min(true_ground_points[:, 0].min(), pred_ground_points[:, 0].min())
    x_max = max(true_ground_points[:, 0].max(), pred_ground_points[:, 0].max())
    y_min = min(true_ground_points[:, 1].min(), pred_ground_points[:, 1].min())
    y_max = max(true_ground_points[:, 1].max(), pred_ground_points[:, 1].max())
    nx = int(np.ceil((x_max - x_min) / resolution)) + 1
    ny = int(np.ceil((y_max - y_min) / resolution)) + 1

    def create_dtm_fast(pts, x_min, y_min, resolution, nx, ny):
        ix = np.clip(((pts[:, 0] - x_min) / resolution).astype(np.int32), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - y_min) / resolution).astype(np.int32), 0, ny - 1)
        z = pts[:, 2]
        dtm = np.full((ny, nx), np.inf, dtype=np.float64)
        np.minimum.at(dtm, (iy, ix), z)
        dtm[dtm == np.inf] = np.nan
        return dtm

    true_dtm = create_dtm_fast(true_ground_points, x_min, y_min, resolution, nx, ny)
    pred_dtm = create_dtm_fast(pred_ground_points, x_min, y_min, resolution, nx, ny)
    original_coverage = (~np.isnan(true_dtm) & ~np.isnan(pred_dtm)).sum() / true_dtm.size

    def interpolate_dtm(dtm, x_min, y_min, resolution):
        valid_mask = ~np.isnan(dtm)
        if valid_mask.sum() == 0:
            return dtm
        y_coords, x_coords = np.where(valid_mask)
        z_values = dtm[valid_mask]
        x_real = x_coords * resolution + x_min
        y_real = y_coords * resolution + y_min
        interpolator = NearestNDInterpolator(np.column_stack([x_real, y_real]), z_values)
        missing_mask = np.isnan(dtm)
        if missing_mask.sum() == 0:
            return dtm
        y_missing, x_missing = np.where(missing_mask)
        dtm_filled = dtm.copy()
        dtm_filled[missing_mask] = interpolator(
            x_missing * resolution + x_min, y_missing * resolution + y_min
        )
        return dtm_filled

    if np.isnan(true_dtm).sum() > 0:
        true_dtm = interpolate_dtm(true_dtm, x_min, y_min, resolution)
    if np.isnan(pred_dtm).sum() > 0:
        pred_dtm = interpolate_dtm(pred_dtm, x_min, y_min, resolution)
    valid_mask = ~np.isnan(true_dtm) & ~np.isnan(pred_dtm)
    if valid_mask.sum() == 0:
        return np.nan, 0.0
    diff = true_dtm[valid_mask] - pred_dtm[valid_mask]
    return float(np.sqrt(np.mean(diff ** 2))), float(original_coverage)


def accumulate_pred_delta_z(pred_global_z_pool, pred_global_z_count,
                             point_indices_batch, pred_delta_z_batch,
                             block_z_mins, block_z_maxs, block_z_centers):
    for i in range(point_indices_batch.shape[0]):
        point_indices = point_indices_batch[i]
        delta_z = pred_delta_z_batch[i].astype(np.float64)
        z_min = float(block_z_mins[i])
        z_max = float(block_z_maxs[i])
        z_range = z_max - z_min
        z_center = float(block_z_centers[i])

        global_z = delta_z * z_range + z_min + z_center

        np.add.at(pred_global_z_pool, point_indices, global_z)
        np.add.at(pred_global_z_count, point_indices, 1)

    return pred_global_z_pool, pred_global_z_count


def calculate_dger_point_rmse(pred_global_z_pool, pred_global_z_count,
                               points, true_labels, ground_class_id=0):
    valid_count_mask = pred_global_z_count > 0
    if valid_count_mask.sum() == 0:
        return np.nan, np.nan

    avg_pred_global_z = np.zeros(len(pred_global_z_pool), dtype=np.float64)
    avg_pred_global_z[valid_count_mask] = (
        pred_global_z_pool[valid_count_mask] / pred_global_z_count[valid_count_mask]
    )

    gt_global_z = points[:, 2].astype(np.float64)

    ground_mask = (true_labels == ground_class_id) & valid_count_mask
    if ground_mask.sum() == 0:
        return np.nan, np.nan

    diff = avg_pred_global_z[ground_mask] - gt_global_z[ground_mask]
    rmse_meters = float(np.sqrt(np.mean(diff ** 2)))

    z_range = gt_global_z.max() - gt_global_z.min()
    rmse_normalized = rmse_meters / (z_range + 1e-8)

    return rmse_normalized, rmse_meters


def test_single_file(test_file, classifier, args, logger, device, NUM_CLASSES):
    def log_string(message):
        logger.info(message)
        print(message)

    visual_dir = Path('log/sem_seg/') / args.log_dir / f'visual_output_{Path(test_file).stem}'
    visual_dir.mkdir(parents=True, exist_ok=True)

    log_string('\n' + '=' * 80)
    log_string(f' Testing: {Path(test_file).name}')
    log_string('=' * 80)
    log_string(' Model: TSRNet V3 (BTMamba + TCAS + DGER)')
    log_string(' DGER RMSE: 修复版v3（全局坐标系对齐，pred_global_z vs points[:,2]）')

    if args.adaptive_features:
        log_string(f' 自适应特征: ENABLED')
        log_string(f'   Density Radius: [{args.density_radius_min}, {args.density_radius_max}] m')
        log_string(f'   Normal K: [{args.normal_k_extreme_sparse}~{args.normal_k_extreme_dense}]')
    else:
        log_string(f' 固定特征: k={args.normal_k}, radius={args.density_radius}')

    log_string(f' BTMamba: d_state={args.mamba_d_state}, '
               f'd_conv={args.mamba_d_conv}, expand={args.mamba_expand}')
    log_string(f' block_size={args.block_size}m, stride={args.stride}m, '
               f'kp_radius_scale={args.kp_radius_scale}')

    try:
        full_scene_data = np.load(test_file, mmap_mode='r')
        points = full_scene_data[:, :3]
        labels = full_scene_data[:, 3]
        log_string(f'✓ 加载 {len(points):,} 个点')
        z_range_scene = points[:, 2].max() - points[:, 2].min()
        log_string(f'  场景高程范围: {z_range_scene:.2f} m '
                   f'(z_min={points[:,2].min():.2f}, z_max={points[:,2].max():.2f})')
    except Exception as e:
        log_string(f"❌ 加载失败: {e}")
        return None

    coord_min, coord_max = np.amin(points, axis=0), np.amax(points, axis=0)

    vote_label_pool = np.zeros((points.shape[0], NUM_CLASSES))

    pred_global_z_pool = np.zeros(points.shape[0], dtype=np.float64)
    pred_global_z_count = np.zeros(points.shape[0], dtype=np.int32)

    scene_analysis = {
        'natural_scores': [],
        'urban_scores': [],
        'uncertainty_scores': [],
        'ground_confidence_scores': [],
        'batch_pred_global_z_mean': [],
        'batch_pred_global_z_std': []
    }

    first_vote_time = None

    for vote_idx in tqdm(range(args.num_votes), total=args.num_votes, desc="Voting Rounds"):
        vote_start_time = time.time()

        if vote_idx > 0 and first_vote_time:
            remaining = args.num_votes - vote_idx
            log_string(f"⏱ 预计剩余: {first_vote_time * remaining / 60:.1f} 分钟")

        manager = Manager()
        block_queue = manager.Queue(maxsize=args.prefetch_batches * args.batch_size * 2)
        batch_queue = manager.Queue(maxsize=args.prefetch_batches)

        producer = Process(target=data_producer,
                           args=(block_queue, batch_queue, points,
                                 coord_min, coord_max, args, vote_idx))
        producer.start()

        workers = []
        for _ in range(args.num_workers):
            worker = Process(target=feature_worker,
                             args=(block_queue, batch_queue, args.batch_size, args))
            worker.start()
            workers.append(worker)

        processed_batches = 0
        active_workers = args.num_workers

        pbar = tqdm(desc=f"Vote {vote_idx + 1}/{args.num_votes}",
                    unit="batch",
                    bar_format='{desc}: {n_fmt} batches [{elapsed}, {rate_fmt}]')

        while active_workers > 0:
            try:
                batch_data = batch_queue.get(timeout=10)

                if batch_data is None:
                    active_workers -= 1
                    continue

                features_batch, indices_batch, batch_z_mins, batch_z_maxs, batch_z_centers = batch_data

                torch_data = torch.from_numpy(features_batch).to(device).float()
                torch_data = torch_data.transpose(2, 1)

                with torch.no_grad():
                    outputs = classifier(torch_data, gt_labels=None)

                    if isinstance(outputs, dict):
                        seg_pred = outputs['output']
                        batch_pred_label = seg_pred.cpu().data.argmax(dim=2).numpy()

                        if (args.compute_dger_rmse and
                                'pred_delta_z' in outputs and
                                outputs['pred_delta_z'] is not None):
                            batch_pred_dz = outputs['pred_delta_z'].cpu().numpy()

                            accumulate_pred_delta_z(
                                pred_global_z_pool, pred_global_z_count,
                                indices_batch, batch_pred_dz,
                                batch_z_mins, batch_z_maxs, batch_z_centers
                            )

                            if processed_batches % 20 == 0:
                                z_ranges_batch = batch_z_maxs - batch_z_mins
                                global_z_approx = (
                                    batch_pred_dz *
                                    z_ranges_batch[:, np.newaxis] +
                                    batch_z_mins[:, np.newaxis] +
                                    batch_z_centers[:, np.newaxis]
                                )
                                scene_analysis['batch_pred_global_z_mean'].append(
                                    float(global_z_approx.mean())
                                )
                                scene_analysis['batch_pred_global_z_std'].append(
                                    float(global_z_approx.std())
                                )

                        if args.analyze_scene and processed_batches % 20 == 0:
                            if 'scene_weights' in outputs:
                                sw = outputs['scene_weights'].cpu().numpy()
                                scene_analysis['natural_scores'].extend(sw[:, 0].tolist())
                                scene_analysis['urban_scores'].extend(sw[:, 1].tolist())

                        if args.monitor_uncertainty and processed_batches % 20 == 0:
                            if 'uncertainty' in outputs:
                                scene_analysis['uncertainty_scores'].append(
                                    outputs['uncertainty'].mean().item()
                                )
                            if 'ground_confidence' in outputs:
                                scene_analysis['ground_confidence_scores'].append(
                                    outputs['ground_confidence'].mean().item()
                                )
                    else:
                        seg_pred = outputs
                        batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                add_vote(vote_label_pool, indices_batch, batch_pred_label)
                processed_batches += 1
                pbar.update(1)

            except queue.Empty:
                if not any(w.is_alive() for w in workers):
                    break
                continue
            except Exception as e:
                log_string(f"❌ 处理错误: {e}")
                import traceback
                traceback.print_exc()
                break

        pbar.close()
        producer.join(timeout=10)
        for worker in workers:
            worker.join(timeout=10)

        vote_elapsed_time = time.time() - vote_start_time
        if vote_idx == 0:
            first_vote_time = vote_elapsed_time
            log_string(f"⏱ 预计总时间: {first_vote_time * args.num_votes / 60:.1f} 分钟")

    pred_labels = np.argmax(vote_label_pool, 1)

    if args.analyze_scene and scene_analysis['natural_scores']:
        log_string("\n" + "=" * 80)
        log_string(" SCENE ANALYSIS")
        log_string("=" * 80)
        avg_natural = np.mean(scene_analysis['natural_scores'])
        avg_urban = np.mean(scene_analysis['urban_scores'])
        log_string(f"  Natural Terrain Score:  {avg_natural:.3f}")
        log_string(f"  Urban Environment Score: {avg_urban:.3f}")
        if avg_natural > 0.6:
            scene_type = "Natural Terrain Dominated"
        elif avg_urban > 0.6:
            scene_type = "Urban Environment Dominated"
        else:
            scene_type = "Mixed Scene"
        log_string(f"  Scene Type: {scene_type}")
    else:
        scene_type = "Unknown"

    if args.monitor_uncertainty and scene_analysis['uncertainty_scores']:
        log_string(f"\n  Avg Uncertainty:       {np.mean(scene_analysis['uncertainty_scores']):.4f}")
        log_string(f"  Avg Ground Confidence: {np.mean(scene_analysis['ground_confidence_scores']):.4f}")

    dger_point_rmse = np.nan
    dger_point_rmse_meters = np.nan

    if args.compute_dger_rmse and not args.skip_rmse:
        log_string("\n" + "=" * 80)
        log_string(" V3 DGER POINT-LEVEL RMSE（修复版v3：全局坐标系对齐）")
        log_string("=" * 80)
        covered_points = (pred_global_z_count > 0).sum()
        log_string(f"  有效高程预测点: {covered_points:,} / {len(points):,} "
                   f"({covered_points/len(points)*100:.1f}%)")

        if scene_analysis['batch_pred_global_z_mean']:
            log_string(f"  预测全局高程均值: "
                       f"{np.mean(scene_analysis['batch_pred_global_z_mean']):.4f} m")
            log_string(f"  场景真实高程均值: {points[:, 2].mean():.4f} m  ← 两者应接近")
            log_string(f"  预测全局高程标准差: "
                       f"{np.mean(scene_analysis['batch_pred_global_z_std']):.4f} m")

        dger_point_rmse, dger_point_rmse_meters = calculate_dger_point_rmse(
            pred_global_z_pool, pred_global_z_count,
            points, labels,
            ground_class_id=0
        )

        if not np.isnan(dger_point_rmse_meters):
            log_string(f"  ✓ DGER点级RMSE: {dger_point_rmse_meters:.4f} 米  "
                       f"(归一化={dger_point_rmse:.6f})")
            log_string(f"    → 含义: 地面点绝对高程预测误差 {dger_point_rmse_meters*100:.2f} cm")
        else:
            log_string("  ⚠️ DGER RMSE计算失败（无有效地面点预测）")

    rmse_dtm, coverage = np.nan, 0.0
    if not args.skip_rmse:
        log_string("\n" + "=" * 80)
        log_string(" DTM QUALITY ASSESSMENT")
        log_string("=" * 80)
        rmse_dtm, coverage = calculate_rmse_dtm(
            points, labels, pred_labels, resolution=args.dtm_resolution
        )
        if not np.isnan(rmse_dtm):
            log_string(f"✓ DTM-RMSE (resolution={args.dtm_resolution}m): {rmse_dtm:.4f} 米")
            log_string(f"✓ DTM覆盖率: {coverage * 100:.1f}%")
        else:
            log_string("⚠️ DTM-RMSE计算失败")

    total_seen_class = np.zeros(NUM_CLASSES, dtype=np.float32)
    total_correct_class = np.zeros(NUM_CLASSES, dtype=np.float32)
    total_iou_deno_class = np.zeros(NUM_CLASSES, dtype=np.float32)
    total_pred_class = np.zeros(NUM_CLASSES, dtype=np.float32)

    for l in range(NUM_CLASSES):
        total_seen_class[l] = np.sum(labels == l)
        total_correct_class[l] = np.sum((pred_labels == l) & (labels == l))
        total_iou_deno_class[l] = np.sum((pred_labels == l) | (labels == l))
        total_pred_class[l] = np.sum(pred_labels == l)

    iou_per_class = total_correct_class / (total_iou_deno_class + 1e-6)
    accuracy_per_class = total_correct_class / (total_seen_class + 1e-6)
    mIoU = np.mean(iou_per_class)
    mAcc = np.mean(accuracy_per_class)
    OA = np.sum(total_correct_class) / (np.sum(total_seen_class) + 1e-6)

    f1_metrics = calculate_f1_scores(
        total_correct_class, total_seen_class, total_pred_class,
        NUM_CLASSES, classes
    )

    log_string('\n' + '=' * 80)
    log_string(' EVALUATION RESULTS - TSRNet V3')
    log_string('=' * 80)
    log_string(f'Overall Accuracy (OA):    {OA:.4f}')
    log_string(f'Mean Accuracy (mAcc):     {mAcc:.4f}')
    log_string(f'Mean IoU (mIoU):          {mIoU:.4f}')
    log_string(f'Macro F1-Score:           {f1_metrics["macro_f1"]:.4f}')
    if not np.isnan(dger_point_rmse_meters):
        log_string(f'DGER点级RMSE:             {dger_point_rmse_meters:.4f} m  ← v3修复后真实值')
    if not np.isnan(rmse_dtm):
        log_string(f'DTM栅格化RMSE:            {rmse_dtm:.4f} m')
    log_string('=' * 80)
    for l in range(NUM_CLASSES):
        class_name = seg_label_to_cat.get(l, f"Class_{l}")
        log_string(f'  {class_name:<12s}: IoU={iou_per_class[l]:.4f}  '
                   f'Acc={accuracy_per_class[l]:.4f}  '
                   f'F1={f1_metrics["class_f1"][l]:.4f}  '
                   f'P={f1_metrics["class_precision"][l]:.4f}  '
                   f'R={f1_metrics["class_recall"][l]:.4f}')

    results = {
        'test_file': str(test_file),
        'model': 'TSRNet_V3',
        'model_version': 'V3',
        'dger_rmse_fix': 'v3_global_coordinate_aligned',
        'innovations': ['BTMamba', 'TCAS', 'DGER'],
        'adaptive_features': args.adaptive_features,
        'mamba_params': {
            'mamba_d_state': args.mamba_d_state,
            'mamba_d_conv': args.mamba_d_conv,
            'mamba_expand': args.mamba_expand,
        },
        'parameters': {
            'density_radius_min': args.density_radius_min,
            'density_radius_max': args.density_radius_max,
            'normal_k_extreme_sparse': args.normal_k_extreme_sparse,
            'normal_k_sparse': args.normal_k_sparse,
            'normal_k_normal': args.normal_k_normal,
            'normal_k_dense': args.normal_k_dense,
            'normal_k_extreme_dense': args.normal_k_extreme_dense,
            'block_size': args.block_size,
            'stride': args.stride,
            'kp_radius_scale': args.kp_radius_scale,
            'num_votes': args.num_votes
        },
        'overall_metrics': {
            'OA': float(OA),
            'mAcc': float(mAcc),
            'mIoU': float(mIoU),
            'macro_f1': float(f1_metrics['macro_f1']),
            'micro_f1': float(f1_metrics['micro_f1']),
            'dger_point_rmse_normalized': float(dger_point_rmse) if not np.isnan(dger_point_rmse) else None,
            'dger_point_rmse_meters': float(dger_point_rmse_meters) if not np.isnan(dger_point_rmse_meters) else None,
            'dtm_rmse_meters': float(rmse_dtm) if not np.isnan(rmse_dtm) else None,
            'dtm_coverage': float(coverage)
        },
        'class_wise_metrics': {
            seg_label_to_cat[l]: {
                'iou': float(iou_per_class[l]),
                'accuracy': float(accuracy_per_class[l]),
                'f1_score': float(f1_metrics['class_f1'][l]),
                'precision': float(f1_metrics['class_precision'][l]),
                'recall': float(f1_metrics['class_recall'][l])
            }
            for l in range(NUM_CLASSES)
        }
    }

    if args.analyze_scene and scene_analysis['natural_scores']:
        results['scene_analysis'] = {
            'natural_terrain_score': float(np.mean(scene_analysis['natural_scores'])),
            'urban_environment_score': float(np.mean(scene_analysis['urban_scores'])),
            'scene_type': scene_type
        }
    if args.monitor_uncertainty and scene_analysis['uncertainty_scores']:
        results['model_diagnostics'] = {
            'avg_uncertainty': float(np.mean(scene_analysis['uncertainty_scores'])),
            'avg_ground_confidence': float(np.mean(scene_analysis['ground_confidence_scores']))
        }
    if scene_analysis['batch_pred_global_z_mean']:
        results['dger_diagnostics'] = {
            'avg_pred_global_z_meters': float(np.mean(scene_analysis['batch_pred_global_z_mean'])),
            'avg_pred_global_z_std_meters': float(np.mean(scene_analysis['batch_pred_global_z_std'])),
            'scene_true_z_mean': float(points[:, 2].mean()),
            'coverage_ratio': float((pred_global_z_count > 0).sum() / len(points))
        }

    if args.save_detailed_results:
        results_path = visual_dir / f'{Path(test_file).stem}_results_v3_fixed.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log_string(f"\n💾 结果保存至: {results_path}")

    if args.visual:
        pred_colors = np.array([g_label2color.get(l, [0, 0, 0]) for l in pred_labels])
        output_data = np.hstack([
            points, pred_colors,
            pred_labels[:, np.newaxis],
            labels[:, np.newaxis]
        ])
        output_filename = visual_dir / f'{Path(test_file).stem}_pred_v3.txt'
        np.savetxt(output_filename, output_data, fmt='%.3f %.3f %.3f %d %d %d %d %d')
        log_string(f"🎨 可视化保存至: {output_filename}")

        if args.compute_dger_rmse and (pred_global_z_count > 0).sum() > 0:
            avg_pred_global_z_vis = np.zeros(len(points))
            valid = pred_global_z_count > 0
            avg_pred_global_z_vis[valid] = pred_global_z_pool[valid] / pred_global_z_count[valid]
            elev_vis_data = np.column_stack([points, avg_pred_global_z_vis, labels])
            elev_vis_path = visual_dir / f'{Path(test_file).stem}_elevation_pred_v3_fixed.txt'
            np.savetxt(elev_vis_path, elev_vis_data,
                       fmt='%.3f %.3f %.3f %.6f %d',
                       header='X Y Z pred_global_z true_label')
            log_string(f"📊 高程预测数据保存至: {elev_vis_path}")

    if args.save_predictions:
        avg_pred_global_z_save = np.zeros(len(points))
        valid = pred_global_z_count > 0
        avg_pred_global_z_save[valid] = pred_global_z_pool[valid] / pred_global_z_count[valid]
        pred_output = np.column_stack([points, pred_labels, avg_pred_global_z_save])
        pred_filename = visual_dir / f'{Path(test_file).stem}_predictions_v3.npy'
        np.save(pred_filename, pred_output)
        log_string(f"💾 预测结果保存至: {pred_filename}")

    return results


def main(args):
    set_random_seeds(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = Path('log/sem_seg/') / args.log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("TSRNet_V3_Test")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if logger.hasHandlers():
        logger.handlers.clear()

    timestamp = int(time.time())
    file_handler = logging.FileHandler(
        str(experiment_dir / f'test_v3_fixed_{timestamp}.txt'), encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(message):
        logger.info(message)
        print(message)

    log_string('=' * 80)
    log_string(' TSRNET V3 TESTING（DGER RMSE 修复版v3 — 全局坐标系对齐）')
    log_string(' pred_delta_z → centered_z → global_z，直接与 points[:,2] 比较')
    log_string('=' * 80)
    for arg in vars(args):
        log_string(f'  {arg:30s}: {getattr(args, arg)}')
    log_string('=' * 80)

    NUM_CLASSES = len(classes)

    log_string(f"\n📦 加载模型: {args.model_name}")
    try:
        MODEL = importlib.import_module(args.model_name)
    except ImportError as e:
        log_string(f"❌ 无法导入模型 {args.model_name}: {e}")
        return

    classifier = MODEL.get_model(
        num_classes=NUM_CLASSES,
        in_channels=11,
        kp_radius_scale=args.kp_radius_scale,
        kp_sigma_scale=0.9,
        enable_domain_adapt=False,
        k_transformer=24,
        n_heads_transformer=8,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        tcas_alpha=0.4,
        tcas_beta=0.4,
        tcas_gamma=0.2,
        dger_k_tscl=8,
        dger_grid_size=32
    ).to(device)

    checkpoint_path = experiment_dir / 'checkpoints' / args.checkpoint_name
    if not checkpoint_path.exists():
        log_string(f"❌ Checkpoint不存在: {checkpoint_path}")
        return

    log_string(f"📂 加载checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if 'model_version' in checkpoint:
            log_string(f"✓ 模型版本: {checkpoint['model_version']}")
        if 'best_iou' in checkpoint:
            log_string(f"✓ 训练最佳 mIoU: {checkpoint['best_iou']:.4f}")
        if 'best_rmse' in checkpoint:
            log_string(f"✓ 训练最佳 RMSE: {checkpoint['best_rmse']:.4f} m")
        if 'epoch' in checkpoint:
            log_string(f"✓ 训练Epoch:      {checkpoint['epoch'] + 1}")
        log_string("✓ Checkpoint加载成功")
    except Exception as e:
        log_string(f"❌ 加载checkpoint失败: {e}")
        import traceback
        traceback.print_exc()
        return

    classifier.eval()
    total_params = sum(p.numel() for p in classifier.parameters())
    v3_params = sum(
        p.numel() for name, p in classifier.named_parameters()
        if any(m in name for m in ['btmamba', 'tcas', 'dger'])
    )
    log_string(f"✓ 总参数量: {total_params:,}")
    log_string(f"✓ V3新增模块参数: {v3_params:,} ({v3_params/total_params*100:.1f}%)")

    test_files = []
    if args.test_all:
        test_dir = Path(args.test_file).parent
        test_files = sorted(list(test_dir.glob('*.npy')))
        log_string(f"\n✓ 测试目录内所有文件: {len(test_files)} 个")
    else:
        test_files = [args.test_file]
        log_string(f"\n✓ 测试单文件: {args.test_file}")

    all_results = []
    for test_file in test_files:
        result = test_single_file(test_file, classifier, args, logger, device, NUM_CLASSES)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        log_string('\n' + '=' * 80)
        log_string(' 所有测试文件汇总')
        log_string('=' * 80)
        avg_miou = np.mean([r['overall_metrics']['mIoU'] for r in all_results])
        avg_oa = np.mean([r['overall_metrics']['OA'] for r in all_results])
        avg_f1 = np.mean([r['overall_metrics']['macro_f1'] for r in all_results])
        dger_rmse_vals = [
            r['overall_metrics']['dger_point_rmse_meters']
            for r in all_results
            if r['overall_metrics'].get('dger_point_rmse_meters') is not None
        ]
        dtm_rmse_vals = [
            r['overall_metrics']['dtm_rmse_meters']
            for r in all_results
            if r['overall_metrics'].get('dtm_rmse_meters') is not None
        ]
        log_string(f"平均 mIoU:           {avg_miou:.4f}")
        log_string(f"平均 OA:             {avg_oa:.4f}")
        log_string(f"平均 Macro F1:       {avg_f1:.4f}")
        if dger_rmse_vals:
            log_string(f"平均 DGER点级RMSE:   {np.mean(dger_rmse_vals):.4f} 米（v3修复后真实值）")
        if dtm_rmse_vals:
            log_string(f"平均 DTM栅格RMSE:    {np.mean(dtm_rmse_vals):.4f} 米")

        log_string('\n逐文件结果:')
        for r in all_results:
            fname = Path(r['test_file']).name
            m = r['overall_metrics']
            dger_r = f"{m['dger_point_rmse_meters']:.4f}m" if m.get('dger_point_rmse_meters') else "N/A"
            dtm_r  = f"{m['dtm_rmse_meters']:.4f}m"        if m.get('dtm_rmse_meters')        else "N/A"
            log_string(f"  {fname:45s} mIoU={m['mIoU']:.4f}  DGER-RMSE={dger_r}  DTM-RMSE={dtm_r}")

        summary_path = experiment_dir / f'test_summary_v3_fixed_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'model': 'TSRNet_V3',
                'dger_rmse_fix': 'v3_global_coordinate_aligned',
                'average_metrics': {
                    'mIoU': float(avg_miou),
                    'OA': float(avg_oa),
                    'macro_f1': float(avg_f1),
                    'dger_point_rmse_meters': float(np.mean(dger_rmse_vals)) if dger_rmse_vals else None,
                    'dtm_rmse_meters': float(np.mean(dtm_rmse_vals)) if dtm_rmse_vals else None,
                },
                'all_results': all_results
            }, f, indent=2)
        log_string(f"\n💾 汇总保存至: {summary_path}")

    log_string('\n' + '=' * 80)
    log_string(' ✅ 测试完成')
    log_string('\n【DGER RMSE Bug修复汇总（v3最终版）】')
    log_string('  1. compute_features_for_block: 新增返回 block_z_center（原始全局z均值）')
    log_string('  2. feature_worker: batch元组从4个扩展为5个，透传z_centers')
    log_string('  3. accumulate_pred_delta_z: global_z = delta_z*range + z_min + z_center')
    log_string('  4. calculate_dger_point_rmse: gt直接用 points[:,2]，无任何中心化')
    log_string('=' * 80)


if __name__ == '__main__':
    args = parse_args()
    if args.model_name.endswith(".py"):
        args.model_name = args.model_name[:-3]
    main(args)