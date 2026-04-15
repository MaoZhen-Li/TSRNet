import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import sys
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

try:
   from scipy.spatial import cKDTree
   from sklearn.neighbors import NearestNeighbors
except ImportError:
   print("❌ 错误: 需要安装scipy和sklearn库")
   print("   请运行: pip install scipy scikit-learn")
   sys.exit(1)


class OpenGFDatasetV3(Dataset):

   def __init__(self,
                data_root='data',
                split='train',
                num_point=8192,
                block_size=32.0,
                samples_per_epoch=10000,
                transform=None,
                density_radius=1.0,
                normal_k=20,
                adaptive_features=True,
                density_radius_min=0.5,
                density_radius_max=5.0,
                normal_k_extreme_sparse=5,
                normal_k_sparse=8,
                normal_k_normal=20,
                normal_k_dense=32,
                normal_k_extreme_dense=48,
                use_cache=True,
                force_recompute=False,
                robust_augmentation=False,
                test1_augmentation=False,
                test2_augmentation=False,
                test3_augmentation=False,
                return_xyz=True,
                elevation_noise_aug=False,
                elevation_noise_std=0.05,
                verbose=True):
       super().__init__()

       self.num_point = num_point
       self.block_size = block_size
       self.transform = transform
       self.split = split
       self.samples_per_epoch = samples_per_epoch
       self.verbose = verbose

       self.density_radius = density_radius
       self.normal_k = normal_k

       self.adaptive_features = adaptive_features
       self.density_radius_min = density_radius_min
       self.density_radius_max = density_radius_max
       self.k_extreme_sparse = normal_k_extreme_sparse
       self.k_sparse = normal_k_sparse
       self.k_normal = normal_k_normal
       self.k_dense = normal_k_dense
       self.k_extreme_dense = normal_k_extreme_dense

       self.use_cache = use_cache
       self.robust_augmentation = robust_augmentation
       self.test1_augmentation = test1_augmentation
       self.test2_augmentation = test2_augmentation
       self.test3_augmentation = test3_augmentation

       self.return_xyz = return_xyz
       self.elevation_noise_aug = elevation_noise_aug
       self.elevation_noise_std = elevation_noise_std

       self.cache_version = '1.2'
       self.cache_dir = os.path.join(data_root, f'cached_features_robust_v{self.cache_version}')
       os.makedirs(self.cache_dir, exist_ok=True)

       self.data_paths = []
       data_path = os.path.join(data_root, self._get_split_folder(split))

       if split == 'train':
           scene_folders = sorted([
               os.path.join(data_path, d) for d in os.listdir(data_path)
               if d.startswith('S') and os.path.isdir(os.path.join(data_path, d))
           ])
           for scene_folder in scene_folders:
               scene_files = [
                   os.path.join(scene_folder, f)
                   for f in os.listdir(scene_folder) if f.endswith('.npy')
               ]
               self.data_paths.extend(scene_files)
       else:
           self.data_paths = [
               os.path.join(data_path, f)
               for f in os.listdir(data_path) if f.endswith('.npy')
           ]

       if not self.data_paths:
           raise FileNotFoundError(
               f"在路径 {data_path} 中没有为 '{split}' split 找到任何 .npy 文件"
           )

       if self.verbose:
           print(f"✓ 为 '{split}' split 找到 {len(self.data_paths)} 个数据文件")
           if self.return_xyz:
               print(f"✓ V3模式: __getitem__ 返回 (points, labels, xyz_raw)")
           if self.elevation_noise_aug and split == 'train':
               print(f"✓ V3高程噪声增强: ENABLED (std={self.elevation_noise_std}m)")

       if use_cache:
           self._prepare_cached_features(force_recompute)

       self.labelweights = self._calculate_label_weights()
       if self.verbose:
           print(f"✓ '{split}' split 的类别权重: {self.labelweights}")

   def _get_split_folder(self, split):
       if split == 'train':
           return 'OpenGF_train'
       elif split == 'validation':
           return 'Validation'
       elif split == 'test':
           return 'OpenGF_test'
       else:
           raise ValueError(f"Unknown split: {split}")

   def _get_cache_path(self, file_path):
       basename = os.path.basename(file_path).replace(
           '.npy', f'_robust_features_v{self.cache_version}.pkl'
       )
       split_cache_dir = os.path.join(self.cache_dir, self.split)
       os.makedirs(split_cache_dir, exist_ok=True)
       return os.path.join(split_cache_dir, basename)

   def _prepare_cached_features(self, force_recompute=False):
       if self.verbose:
           print(f"检查 '{self.split}' split 的缓存特征（V{self.cache_version}）...")

       files_to_process = []
       for file_path in self.data_paths:
           cache_path = self._get_cache_path(file_path)
           if force_recompute or not os.path.exists(cache_path):
               files_to_process.append(file_path)
           else:
               try:
                   with open(cache_path, 'rb') as f:
                       cache_data = pickle.load(f)
                   if cache_data.get('version', '1.0') != self.cache_version:
                       files_to_process.append(file_path)
               except Exception:
                   files_to_process.append(file_path)

       if not files_to_process:
           if self.verbose:
               print(f"✓ 所有 {len(self.data_paths)} 个文件的特征已缓存（V{self.cache_version}）")
           return

       if self.verbose:
           print(f"需要计算特征的文件: {len(files_to_process)}/{len(self.data_paths)}")

       for file_path in tqdm(
           files_to_process,
           desc=f"预计算 {self.split} 特征 (V{self.cache_version})",
           ascii=True,
           disable=not self.verbose
       ):
           try:
               self._compute_and_cache_features(file_path)
           except Exception as e:
               print(f"\n❌ 警告: 处理文件 {file_path} 时出错: {e}")
               import traceback
               traceback.print_exc()

       if self.verbose:
           print(f"✓ 特征预计算完成（V{self.cache_version}）")

   def _compute_adaptive_k(self, median_dist):
       if median_dist > 3.0:
           return self.k_extreme_sparse, "Extreme Sparse"
       elif median_dist > 1.5:
           return self.k_sparse, "Sparse"
       elif median_dist > 0.8:
           return self.k_normal, "Normal"
       elif median_dist > 0.3:
           return self.k_dense, "Dense"
       else:
           return self.k_extreme_dense, "Extreme Dense"

   def _compute_normals_and_curvature_adaptive(self, points, median_dist=None, verbose=False):
       n_points = points.shape[0]
       normals = np.zeros((n_points, 3), dtype=np.float32)
       curvatures = np.zeros(n_points, dtype=np.float32)

       if median_dist is None:
           sample_size = min(1000, n_points)
           sample_idx = np.random.choice(n_points, sample_size, replace=False)
           tree = cKDTree(points)
           try:
               distances, _ = tree.query(points[sample_idx], k=20)
               median_dist = np.median(distances[:, -1])
           except Exception:
               median_dist = 1.0

       k, scene_type = self._compute_adaptive_k(median_dist)
       k = min(k, n_points)

       if verbose:
           print(f"  └─ 自适应k={k} ({scene_type}, median_dist={median_dist:.3f}m)")

       try:
           nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(points)
           distances, indices = nbrs.kneighbors(points)
       except Exception as e:
           if verbose:
               print(f"  └─ KNN失败，使用默认法向量: {e}")
           normals[:, 2] = 1.0
           return normals, curvatures

       for i in range(n_points):
           try:
               neighbor_indices = indices[i]
               neighbors = points[neighbor_indices]
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

           except Exception:
               normals[i] = np.array([0, 0, 1], dtype=np.float32)
               curvatures[i] = 0.0

       return normals, curvatures

   def _compute_normals_and_curvature_fixed(self, points):
       n_points = points.shape[0]
       normals = np.zeros((n_points, 3), dtype=np.float32)
       curvatures = np.zeros(n_points, dtype=np.float32)
       k = min(self.normal_k, n_points)

       try:
           nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(points)
           _, indices = nbrs.kneighbors(points)
       except Exception:
           normals[:, 2] = 1.0
           return normals, curvatures

       for i in range(n_points):
           try:
               neighbors = points[indices[i]]
               centroid = neighbors.mean(axis=0)
               centered = neighbors - centroid
               cov = np.dot(centered.T, centered) / k
               eigenvalues, eigenvectors = np.linalg.eigh(cov)
               normal = eigenvectors[:, 0]
               if normal[2] < 0:
                   normal = -normal
               normals[i] = normal
               eigenvalues = np.sort(eigenvalues)
               curvatures[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-8)
           except Exception:
               normals[i] = np.array([0, 0, 1], dtype=np.float32)
               curvatures[i] = 0.0

       return normals, curvatures

   def _compute_and_cache_features(self, file_path):
       verbose = self.verbose and (np.random.rand() < 0.05)

       room_data = np.load(file_path)
       points = room_data[:, :3].astype(np.float32)
       labels = room_data[:, 3].astype(np.int64)

       tree = cKDTree(points)
       median_dist = None

       if self.adaptive_features:
           sample_size = min(1000, len(points))
           sample_idx = np.random.choice(len(points), sample_size, replace=False)
           try:
               distances, _ = tree.query(points[sample_idx], k=20)
               median_dist = np.median(distances[:, -1])
               adaptive_radius = np.clip(
                   median_dist * 3,
                   self.density_radius_min,
                   self.density_radius_max
               )
               density = tree.query_ball_point(
                   points, r=adaptive_radius, return_length=True
               ).astype(np.float32)
               max_density = np.percentile(density, 95)
               density = np.clip(
                   density / max_density if max_density > 0 else density / 100.0,
                   0, 1.0
               )
           except Exception:
               density = tree.query_ball_point(
                   points, r=self.density_radius, return_length=True
               ).astype(np.float32)
               density = np.clip(density / 100.0, 0, 1.0)
               median_dist = None
       else:
           density = tree.query_ball_point(
               points, r=self.density_radius, return_length=True
           ).astype(np.float32)
           density = np.clip(density / 100.0, 0, 1.0)

       if self.adaptive_features:
           normals, curvatures = self._compute_normals_and_curvature_adaptive(
               points, median_dist, verbose=verbose
           )
       else:
           normals, curvatures = self._compute_normals_and_curvature_fixed(points)

       cache_data = {
           'points': points,
           'labels': labels,
           'normals': normals,
           'curvatures': curvatures,
           'density': density,
           'version': self.cache_version,
           'adaptive_features': self.adaptive_features,
           'median_dist': median_dist,
           'timestamp': datetime.now().isoformat()
       }

       cache_path = self._get_cache_path(file_path)
       with open(cache_path, 'wb') as f:
           pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

   def _calculate_label_weights(self):
       if self.verbose:
           print(f"正在为 '{self.split}' split 计算类别权重...")

       labelweights = np.zeros(2, dtype=np.float32)
       for file_path in tqdm(
           self.data_paths, desc="计算权重", leave=False,
           ascii=True, disable=not self.verbose
       ):
           try:
               if self.use_cache:
                   cache_path = self._get_cache_path(file_path)
                   if os.path.exists(cache_path):
                       with open(cache_path, 'rb') as f:
                           cache_data = pickle.load(f)
                       labels = cache_data['labels']
                   else:
                       labels = np.load(file_path)[:, 3]
               else:
                   labels = np.load(file_path)[:, 3]

               tmp, _ = np.histogram(labels, range(3))
               labelweights += tmp[:2]
           except Exception as e:
               print(f"\n警告: 计算权重时出错 {file_path}: {e}")

       labelweights = labelweights.astype(np.float32)
       total = np.sum(labelweights)
       if total == 0:
           return np.ones(2, dtype=np.float32)

       labelweights = labelweights / total
       weights = np.power(np.amax(labelweights) / (labelweights + 1e-6), 1 / 3.0)
       return weights

   def _apply_test1_augmentation(self, points, labels):
       coords = points[:, :3].copy()
       N = coords.shape[0]
       if np.random.rand() > 0.5:
           transition_ratio = 0.3
           num_transition = int(N * transition_ratio)
           transition_idx = np.random.choice(N, num_transition, replace=False)
           height_change = np.random.choice([-1, 1]) * np.random.uniform(3, 15)
           coords[transition_idx, 2] += height_change * np.random.rand(num_transition)
       if np.random.rand() > 0.6:
           slope_noise = np.random.uniform(0.05, 0.15)
           x_gradient = coords[:, 0] - coords[:, 0].mean()
           y_gradient = coords[:, 1] - coords[:, 1].mean()
           coords[:, 2] += (x_gradient + y_gradient) * slope_noise
       if np.random.rand() > 0.5:
           micro_scale = np.random.uniform(0.02, 0.08)
           coords[:, 2] += np.random.randn(N).astype(np.float32) * micro_scale
       points[:, :3] = coords
       return points, labels

   def _apply_test2_augmentation(self, points, labels):
       coords = points[:, :3].copy()
       N = coords.shape[0]
       if np.random.rand() > 0.2:
           outlier_ratio = np.random.uniform(0.15, 0.25)
           num_outliers = int(N * outlier_ratio)
           outlier_idx = np.random.choice(N, num_outliers, replace=False)
           coords[outlier_idx, 2] += (
               np.random.choice([-1, 1], num_outliers) *
               np.random.uniform(20, 50, num_outliers)
           )
       if np.random.rand() > 0.3:
           plane_ratio = np.random.uniform(0.4, 0.6)
           num_plane = int(N * plane_ratio)
           plane_idx = np.random.choice(N, num_plane, replace=False)
           plane_height = np.random.uniform(10, 40)
           coords[plane_idx, 2] = plane_height + np.random.randn(num_plane) * 0.15
       if np.random.rand() > 0.4:
           noise_level = np.random.uniform(0.05, 0.12)
           coords += np.random.randn(*coords.shape).astype(np.float32) * noise_level
       if np.random.rand() > 0.5:
           center = coords.mean(axis=0)
           distances = np.linalg.norm(coords - center, axis=1)
           far_mask = distances > np.percentile(distances, 50)
           dropout_mask = np.random.rand(N) < 0.4
           keep_mask = ~(far_mask & dropout_mask)
           if keep_mask.sum() > N // 2:
               coords = coords[keep_mask]
               points = points[keep_mask]
               labels = labels[keep_mask]
               shortage = N - coords.shape[0]
               if shortage > 0:
                   supplement_idx = np.random.choice(coords.shape[0], shortage, replace=True)
                   points = np.vstack([points, points[supplement_idx]])
                   labels = np.concatenate([labels, labels[supplement_idx]])
                   coords = points[:, :3]
       points[:, :3] = coords
       return points, labels

   def _apply_test3_augmentation(self, points, labels):
       coords = points[:, :3].copy()
       N = coords.shape[0]
       if np.random.rand() > 0.6:
           num_terraces = np.random.randint(2, 5)
           for _ in range(num_terraces):
               terrace_ratio = np.random.uniform(0.15, 0.30)
               num_terrace = int(N * terrace_ratio)
               terrace_idx = np.random.choice(N, num_terrace, replace=False)
               terrace_height = np.random.uniform(1, 5)
               coords[terrace_idx, 2] = (
                   (coords[terrace_idx, 2] // terrace_height) * terrace_height
               )
       if np.random.rand() > 0.7:
           crack_ratio = 0.2
           num_crack = int(N * crack_ratio)
           crack_idx = np.random.choice(N, num_crack, replace=False)
           crack_depth = np.random.uniform(3, 10)
           coords[crack_idx, 2] -= crack_depth
       if np.random.rand() > 0.5:
           sparse_ratio = np.random.uniform(0.5, 0.7)
           keep_num = int(N * sparse_ratio)
           keep_idx = np.sort(np.random.choice(N, keep_num, replace=False))
           coords = coords[keep_idx]
           points = points[keep_idx]
           labels = labels[keep_idx]
           shortage = N - coords.shape[0]
           if shortage > 0:
               supplement_idx = np.random.choice(coords.shape[0], shortage, replace=True)
               points = np.vstack([points, points[supplement_idx]])
               labels = np.concatenate([labels, labels[supplement_idx]])
               coords = points[:, :3]
       if np.random.rand() > 0.5:
           slope = np.random.uniform(0.1, 0.3)
           direction = np.random.choice(['x', 'y'])
           if direction == 'x':
               coords[:, 2] += coords[:, 0] * slope
           else:
               coords[:, 2] += coords[:, 1] * slope
       points[:, :3] = coords
       return points, labels

   def _apply_robust_augmentation(self, points, labels):
       coords = points[:, :3].copy()
       if np.random.rand() > 0.5:
           angle = np.random.uniform(0, 2 * np.pi)
           cos_a, sin_a = np.cos(angle), np.sin(angle)
           rot_matrix = np.array(
               [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32
           )
           coords = np.dot(coords, rot_matrix.T)
       if np.random.rand() > 0.5:
           coords *= np.random.uniform(0.85, 1.15)
       if np.random.rand() > 0.5:
           coords += np.random.uniform(-0.2, 0.2, size=(1, 3))
       if np.random.rand() > 0.5:
           noise_level = np.random.uniform(0.01, 0.05)
           coords += np.random.randn(*coords.shape).astype(np.float32) * noise_level
       if np.random.rand() > 0.5:
           drop_ratio = np.random.uniform(0.1, 0.2)
           num_points = coords.shape[0]
           keep_num = int(num_points * (1 - drop_ratio))
           keep_idx = np.sort(np.random.choice(num_points, keep_num, replace=False))
           if keep_num < num_points:
               supplement_idx = np.random.choice(keep_num, num_points - keep_num, replace=True)
               keep_idx = np.concatenate([keep_idx, keep_idx[supplement_idx]])
           points = points[keep_idx]
           labels = labels[keep_idx]
           coords = points[:, :3]
       if np.random.rand() > 0.5:
           coords += np.random.normal(0, 0.01, size=coords.shape).astype(np.float32)
       points[:, :3] = coords
       return points, labels

   def _apply_elevation_noise_augmentation(self, points, labels):
       coords = points[:, :3].copy()
       N = coords.shape[0]

       if np.random.rand() > 0.3:
           systematic_bias = np.random.normal(0, self.elevation_noise_std * 0.5)
           coords[:, 2] += systematic_bias

       ground_mask = (labels == 1)
       if np.random.rand() > 0.3:
           ground_noise_std = self.elevation_noise_std * np.random.uniform(0.3, 0.7)
           ground_noise = np.random.normal(0, ground_noise_std, ground_mask.sum())
           coords[ground_mask, 2] += ground_noise.astype(np.float32)

           non_ground_mask = ~ground_mask
           non_ground_noise_std = self.elevation_noise_std * np.random.uniform(0.8, 1.5)
           non_ground_noise = np.random.normal(0, non_ground_noise_std, non_ground_mask.sum())
           coords[non_ground_mask, 2] += non_ground_noise.astype(np.float32)

       if np.random.rand() > 0.5:
           x_coords = coords[:, 0]
           x_threshold = np.percentile(x_coords, np.random.uniform(20, 80))
           strip_mask = x_coords > x_threshold
           strip_bias = np.random.normal(0, self.elevation_noise_std * 1.5)
           coords[strip_mask, 2] += strip_bias

       points[:, :3] = coords
       return points, labels

   def __getitem__(self, index):
       file_path = random.choice(self.data_paths)

       try:
           if self.use_cache:
               cache_path = self._get_cache_path(file_path)
               with open(cache_path, 'rb') as f:
                   cache_data = pickle.load(f)
               points = cache_data['points']
               labels = cache_data['labels']
               normals = cache_data['normals']
               curvatures = cache_data['curvatures']
               density = cache_data['density']
           else:
               room_data = np.load(file_path)
               points = room_data[:, :3].astype(np.float32)
               labels = room_data[:, 3].astype(np.int64)
               tree = cKDTree(points)
               if self.adaptive_features:
                   sample_size = min(1000, len(points))
                   sample_idx = np.random.choice(len(points), sample_size, replace=False)
                   distances, _ = tree.query(points[sample_idx], k=20)
                   median_dist = np.median(distances[:, -1])
                   adaptive_radius = np.clip(
                       median_dist * 3, self.density_radius_min, self.density_radius_max
                   )
                   density = tree.query_ball_point(
                       points, r=adaptive_radius, return_length=True
                   ).astype(np.float32)
                   max_density = np.percentile(density, 95)
                   density = np.clip(
                       density / max_density if max_density > 0 else density / 100.0,
                       0, 1.0
                   )
                   normals, curvatures = self._compute_normals_and_curvature_adaptive(
                       points, median_dist
                   )
               else:
                   density = tree.query_ball_point(
                       points, r=self.density_radius, return_length=True
                   ).astype(np.float32)
                   density = np.clip(density / 100.0, 0, 1.0)
                   normals, curvatures = self._compute_normals_and_curvature_fixed(points)
       except Exception as e:
           print(f"警告: 加载文件 {file_path} 时出错: {e}。重试...")
           return self.__getitem__(index)

       N_points = points.shape[0]

       max_attempts = 50
       for attempt in range(max_attempts):
           center = points[np.random.choice(N_points)][:3]
           block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
           block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
           point_idxs = np.where(
               (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
               (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
           )[0]
           if point_idxs.size > 1024:
               break
           if attempt == max_attempts - 1:
               point_idxs = np.arange(N_points)

       if point_idxs.size >= self.num_point:
           selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
       else:
           selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

       selected_points = points[selected_point_idxs]
       current_labels = labels[selected_point_idxs]
       selected_normals = normals[selected_point_idxs]
       selected_curvatures = curvatures[selected_point_idxs]
       selected_density = density[selected_point_idxs]

       current_points = np.zeros((self.num_point, 11), dtype=np.float32)

       selected_points_centered = selected_points - selected_points.mean(axis=0)
       current_points[:, 0:3] = selected_points_centered

       coord_max = points.max(axis=0)
       coord_min = points.min(axis=0)
       coord_range = coord_max - coord_min
       coord_range[coord_range == 0] = 1.0
       current_points[:, 3] = (selected_points[:, 0] - coord_min[0]) / coord_range[0]
       current_points[:, 4] = (selected_points[:, 1] - coord_min[1]) / coord_range[1]
       current_points[:, 5] = (selected_points[:, 2] - coord_min[2]) / coord_range[2]

       current_points[:, 6] = selected_density
       current_points[:, 7:10] = selected_normals
       current_points[:, 10] = selected_curvatures

       xyz_raw = selected_points_centered.copy()

       if self.split == 'train':
           if self.test1_augmentation and np.random.rand() > 0.4:
               current_points, current_labels = self._apply_test1_augmentation(
                   current_points, current_labels
               )
               xyz_raw = current_points[:, :3].copy()

           if self.test2_augmentation and np.random.rand() > 0.3:
               current_points, current_labels = self._apply_test2_augmentation(
                   current_points, current_labels
               )
               xyz_raw = current_points[:, :3].copy()

           if self.test3_augmentation and np.random.rand() > 0.4:
               current_points, current_labels = self._apply_test3_augmentation(
                   current_points, current_labels
               )
               xyz_raw = current_points[:, :3].copy()

           if self.robust_augmentation:
               current_points, current_labels = self._apply_robust_augmentation(
                   current_points, current_labels
               )
               xyz_raw = current_points[:, :3].copy()

           if self.elevation_noise_aug and np.random.rand() > 0.3:
               current_points, current_labels = self._apply_elevation_noise_augmentation(
                   current_points, current_labels
               )
               xyz_raw = current_points[:, :3].copy()

       if self.transform is not None:
           current_points, current_labels = self.transform(current_points, current_labels)

       if self.return_xyz:
           return (
               current_points,
               current_labels,
               xyz_raw
           )
       else:
           return current_points, current_labels

   def __len__(self):
       return self.samples_per_epoch


def precompute_all_features(data_root='data', adaptive_features=True,
                           density_radius_max=5.0, verbose=True):
   print("=" * 80)
   print("TSRNet V3 数据加载器 V1.3 - 特征预计算")
   print("=" * 80)

   for split in ['train', 'validation', 'test']:
       print(f"\n{'=' * 60}")
       print(f"处理 {split.upper()} split...")
       print('=' * 60)
       try:
           dataset = OpenGFDatasetV3(
               data_root=data_root,
               split=split,
               use_cache=True,
               force_recompute=True,
               adaptive_features=adaptive_features,
               density_radius_max=density_radius_max,
               return_xyz=True,
               verbose=verbose
           )
           print(f"✓ {split} split 完成！共 {len(dataset.data_paths)} 个文件")
       except Exception as e:
           print(f"❌ 处理 {split} split 时出错: {e}")
           import traceback
           traceback.print_exc()

   print("\n" + "=" * 80)
   print("✓ 所有特征预计算完成（V1.3）")
   print("  训练命令: python train_v3.py")
   print("=" * 80)


def verify_dataset_output(data_root='data', adaptive_features=True):
   print("\n" + "=" * 80)
   print("验证数据集输出格式（V1.3 / TSRNet V3）...")
   print("=" * 80)

   dataset = OpenGFDatasetV3(
       data_root=data_root,
       split='train',
       num_point=8192,
       use_cache=True,
       robust_augmentation=True,
       test2_augmentation=True,
       adaptive_features=adaptive_features,
       return_xyz=True,
       elevation_noise_aug=True,
       elevation_noise_std=0.05,
       verbose=True
   )

   result = dataset[0]
   assert len(result) == 3, "V3应返回3个值: (points, labels, xyz_raw)"
   points, labels, xyz_raw = result

   print(f"\n【返回值验证】")
   print(f"  ✓ points  shape: {points.shape}  (期望: ({dataset.num_point}, 11))")
   print(f"  ✓ labels  shape: {labels.shape}  (期望: ({dataset.num_point},))")
   print(f"  ✓ xyz_raw shape: {xyz_raw.shape} (期望: ({dataset.num_point}, 3)) ← V3新增")
   print(f"  ✓ points  dtype: {points.dtype}  (期望: float32)")
   print(f"  ✓ labels  dtype: {labels.dtype}  (期望: int64)")
   print(f"  ✓ xyz_raw dtype: {xyz_raw.dtype} (期望: float32)")
   print(f"  ✓ 标签类别: {np.unique(labels)}")

   print(f"\n【xyz_raw 高程统计（用于DGER）】")
   z = xyz_raw[:, 2]
   print(f"  z_min={z.min():.3f}m, z_max={z.max():.3f}m, z_range={z.max()-z.min():.3f}m")
   print(f"  地面点高程 mean={z[labels==1].mean():.3f}m (若有地面点)")

   print(f"\n【11维特征统计】")
   feature_names = [
       'Center X', 'Center Y', 'Center Z',
       'Norm X',   'Norm Y',   'Norm Z',
       'Density',
       'Normal X', 'Normal Y', 'Normal Z',
       'Curvature'
   ]
   for i, name in enumerate(feature_names):
       feat = points[:, i]
       print(f"  {name:12s}: min={feat.min():.4f}, max={feat.max():.4f}, "
             f"mean={feat.mean():.4f}, std={feat.std():.4f}")

   print("\n✓ V3数据集验证通过！")
   print("=" * 80)


def example_training_loop_v3():
   from torch.utils.data import DataLoader

   train_dataset = OpenGFDatasetV3(
       data_root='data',
       split='train',
       num_point=8192,
       block_size=32.0,
       samples_per_epoch=10000,
       adaptive_features=True,
       robust_augmentation=True,
       test1_augmentation=True,
       test2_augmentation=True,
       test3_augmentation=True,
       return_xyz=True,
       elevation_noise_aug=True,
       elevation_noise_std=0.05,
       verbose=True
   )

   train_loader = DataLoader(
       train_dataset,
       batch_size=8,
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )

   print("【V3训练循环示例（伪代码）】")
   print("""
   for batch_idx, (points, labels, xyz_raw) in enumerate(train_loader):
       points = points.permute(0, 2, 1).cuda()
       labels = labels.cuda()
       xyz    = xyz_raw.cuda()

       pred = model(points, gt_labels=labels)
       loss, loss_dict = loss_fn(pred, labels, xyz=xyz)
       rmse = TSRNetTrainer.compute_rmse(pred['pred_delta_z'], xyz, labels)
   """)


if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(description='OpenGF数据集预处理（V1.3 - TSRNet V3专用）')
   parser.add_argument('--data_root', type=str, default='data')
   parser.add_argument('--precompute', action='store_true')
   parser.add_argument('--verify', action='store_true')
   parser.add_argument('--no_adaptive', action='store_true')
   parser.add_argument('--density_radius_max', type=float, default=5.0)
   parser.add_argument('--quiet', action='store_true')
   args = parser.parse_args()

   adaptive_features = not args.no_adaptive
   verbose = not args.quiet

   if args.precompute:
       precompute_all_features(args.data_root, adaptive_features,
                               args.density_radius_max, verbose)
   if args.verify:
       verify_dataset_output(args.data_root, adaptive_features)
   if not args.precompute and not args.verify:
       example_training_loop_v3()