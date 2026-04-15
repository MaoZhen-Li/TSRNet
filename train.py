import argparse
import os
from dataloader import OpenGFDatasetV3
import torch
import torch.optim as optim
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import random
import json

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

torch.cuda.empty_cache()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ground', 'non-ground']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_label_to_cat = {i: cat for i, cat in enumerate(class2label.keys())}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子设置为: {seed}")


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('TSRNet V3 Training')

    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--log_dir', type=str, default='TSRNet')
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--npoint', type=int, default=8192)
    parser.add_argument('--num_workers', type=int, default=20)

    parser.add_argument('--progressive_training', action='store_true', default=False)
    parser.add_argument('--freeze_epochs', type=int, default=30)
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--robust_augmentation', action='store_true', default=True)
    parser.add_argument('--test1_augmentation', action='store_true', default=False)
    parser.add_argument('--test2_augmentation', action='store_true', default=True)
    parser.add_argument('--test3_augmentation', action='store_true', default=False)
    parser.add_argument('--samples_per_epoch', type=int, default=10500)
    parser.add_argument('--val_samples', type=int, default=1050)

    parser.add_argument('--elevation_noise_aug', action='store_true', default=True)
    parser.add_argument('--elevation_noise_std', type=float, default=0.05)

    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--lambda2', type=float, default=0.3)
    parser.add_argument('--lambda3', type=float, default=0.08)
    parser.add_argument('--lambda4', type=float, default=0.2)

    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)

    parser.add_argument('--save_metric', type=str, default='miou',
                        choices=['miou', 'combined'])
    parser.add_argument('--rmse_weight_in_combined', type=float, default=0.3)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_freq', type=int, default=30)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()


def freeze_domain_layers(model):
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in [
            'domain_align', 'domain_adaptive', 'instance_norm',
            'style_branch', 'domain_invariant'
        ]):
            param.requires_grad = False
            frozen_count += 1
    return frozen_count


def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_rmse_from_xyz(pred_delta_z, xyz, gt_labels, ground_class_id=0):
    import torch.nn.functional as F

    z = xyz[:, :, 2]
    z_min = z.min(dim=1, keepdim=True)[0]
    z_max = z.max(dim=1, keepdim=True)[0]
    gt_delta_z = (z - z_min) / (z_max - z_min + 1e-8)

    ground_mask = (gt_labels == ground_class_id)
    if ground_mask.sum() == 0:
        return 0.0

    mse_norm = F.mse_loss(pred_delta_z[ground_mask], gt_delta_z[ground_mask])

    z_range = (z_max - z_min).mean()
    rmse = (torch.sqrt(mse_norm) * z_range).item()
    return rmse


def evaluate_model(data_loader, model, criterion, num_classes, seg_label_to_cat,
                   desc="Evaluating", use_amp=False, device='cuda',
                   monitor_uncertainty=True):
    model.eval()

    with torch.no_grad():
        num_batches = len(data_loader)
        if num_batches == 0:
            return None

        total_correct, total_seen, loss_sum = 0, 0, 0
        total_seen_class = [0] * num_classes
        total_correct_class = [0] * num_classes
        total_iou_deno_class = [0] * num_classes

        rmse_list = []
        scene_natural_scores = []
        scene_urban_scores = []
        uncertainty_scores = []
        ground_confidence_scores = []
        elev_loss_list = []
        tscl_loss_list = []

        for i, (points, target, xyz_raw) in tqdm(
            enumerate(data_loader), total=num_batches,
            smoothing=0.9, desc=desc, leave=False, ascii=True
        ):
            points = points.float().to(device)
            target = target.long().to(device)
            xyz = xyz_raw.float().to(device)
            points = points.transpose(2, 1)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(points, gt_labels=None)
                    seg_pred = outputs['output']
                    loss, _ = criterion(outputs, target, xyz=xyz)
            else:
                outputs = model(points, gt_labels=None)
                seg_pred = outputs['output']
                loss, _ = criterion(outputs, target, xyz=xyz)

            pred_val = seg_pred.cpu().data.numpy()
            batch_label = target.cpu().data.numpy().reshape(-1)
            loss_sum += loss.item()

            pred_val = np.argmax(pred_val, axis=2).reshape(-1)
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += points.size(0) * points.size(2)

            for l in range(num_classes):
                total_seen_class[l] += np.sum(batch_label == l)
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label == l))

            if 'pred_delta_z' in outputs and outputs['pred_delta_z'] is not None:
                rmse = compute_rmse_from_xyz(
                    outputs['pred_delta_z'],
                    xyz,
                    target,
                    ground_class_id=0
                )
                rmse_list.append(rmse)

            if 'scene_weights' in outputs and i < 20:
                sw = outputs['scene_weights'].cpu().numpy()
                scene_natural_scores.extend(sw[:, 0].tolist())
                scene_urban_scores.extend(sw[:, 1].tolist())

            if monitor_uncertainty and i < 10:
                uncertainty_scores.append(outputs['uncertainty'].mean().item())
                ground_confidence_scores.append(outputs['ground_confidence'].mean().item())

        OA = total_correct / float(total_seen) if total_seen > 0 else 0
        class_accuracies = np.array(total_correct_class) / (
            np.array(total_seen_class, dtype=np.float32) + 1e-6
        )
        MAcc = np.mean(class_accuracies)
        class_ious = np.array(total_correct_class) / (
            np.array(total_iou_deno_class, dtype=np.float32) + 1e-6
        )
        mIoU = np.mean(class_ious)

        metrics = {
            'OA': OA,
            'MAcc': MAcc,
            'mIoU': mIoU,
            'loss': loss_sum / float(num_batches),
            'class_iou': {seg_label_to_cat[l]: class_ious[l] for l in range(num_classes)},
            'class_acc': {seg_label_to_cat[l]: class_accuracies[l] for l in range(num_classes)}
        }

        if rmse_list:
            metrics['rmse'] = np.mean(rmse_list)
            metrics['rmse_std'] = np.std(rmse_list)

        if scene_natural_scores:
            metrics['scene_natural_mean'] = np.mean(scene_natural_scores)
            metrics['scene_urban_mean'] = np.mean(scene_urban_scores)

        if monitor_uncertainty and uncertainty_scores:
            metrics['avg_uncertainty'] = np.mean(uncertainty_scores)
            metrics['avg_ground_confidence'] = np.mean(ground_confidence_scores)

        return metrics


def main(args):
    def log_string(str_):
        logger.info(str_)
        print(str_)

    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = Path('./log/sem_seg/') / args.log_dir
    experiment_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = experiment_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("TSRNet_V3")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir / f'{args.model}.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string('=' * 80)
    log_string('TSRNET V3 - TERRAIN SURFACE RECONSTRUCTION NETWORK')
    log_string('创新点: BTMamba + TCAS + DGER (联合分类-高程回归)')
    log_string('递进关系: V2(场景自适应分类) → V3(分类+地形几何重建联合)')
    log_string('=' * 80)
    log_string('参数配置:')
    for arg in vars(args):
        log_string(f'  {arg:30s}: {getattr(args, arg)}')
    log_string('=' * 80)

    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    DATA_ROOT = 'data'

    log_string("\n--- 加载数据集（OpenGFDatasetV3 V1.3）---")

    log_string("加载训练集...")
    TRAIN_DATASET = OpenGFDatasetV3(
        data_root=DATA_ROOT,
        split='train',
        num_point=NUM_POINT,
        block_size=32.0,
        samples_per_epoch=args.samples_per_epoch,
        use_cache=True,
        force_recompute=False,
        adaptive_features=True,
        density_radius_max=5.0,
        normal_k_extreme_sparse=5,
        normal_k_sparse=8,
        normal_k_dense=32,
        normal_k_extreme_dense=48,
        robust_augmentation=args.robust_augmentation,
        test1_augmentation=args.test1_augmentation,
        test2_augmentation=args.test2_augmentation,
        test3_augmentation=args.test3_augmentation,
        return_xyz=True,
        elevation_noise_aug=args.elevation_noise_aug,
        elevation_noise_std=args.elevation_noise_std,
        verbose=True
    )

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker
    )

    log_string("加载验证集...")
    VALIDATION_DATASET = OpenGFDatasetV3(
        data_root=DATA_ROOT,
        split='validation',
        num_point=NUM_POINT,
        block_size=32.0,
        samples_per_epoch=args.val_samples,
        use_cache=True,
        adaptive_features=True,
        density_radius_max=5.0,
        robust_augmentation=False,
        return_xyz=True,
        elevation_noise_aug=False,
        verbose=True
    )
    valDataLoader = torch.utils.data.DataLoader(
        VALIDATION_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    weights = torch.Tensor(TRAIN_DATASET.labelweights).to(device)

    log_string(f"\n数据集统计:")
    log_string(f"  训练迭代/epoch:   {len(trainDataLoader)} x {BATCH_SIZE} = {len(trainDataLoader)*BATCH_SIZE}")
    log_string(f"  验证迭代/epoch:   {len(valDataLoader)} x {BATCH_SIZE} = {len(valDataLoader)*BATCH_SIZE}")
    log_string(f"  类别权重:         {weights.cpu().numpy()}")
    log_string(f"  特征维度:         11D（与V2完全相同）")
    log_string(f"  V3新增返回值:     xyz_raw (B, N, 3)，供DGER使用")

    log_string(f"\n--- 加载模型: {args.model} ---")

    try:
        MODEL = importlib.import_module(args.model)
        shutil.copy(f'models/{args.model}.py', str(experiment_dir))
    except Exception as e:
        log_string(f"错误: 无法导入模型 {args.model}: {e}")
        return

    model = MODEL.get_model(
        num_classes=NUM_CLASSES,
        in_channels=11,
        kp_radius_scale=6.0,
        kp_sigma_scale=1.0,
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

    criterion = MODEL.TSRNetLoss(
        num_classes=NUM_CLASSES,
        ground_class_id=0,
        lambda1_base=args.lambda1,
        lambda2_base=args.lambda2,
        lambda3=args.lambda3,
        lambda4=args.lambda4
    ).to(device)

    model.apply(inplace_relu)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    v3_module_params = sum(
        p.numel() for name, p in model.named_parameters()
        if any(m in name for m in ['btmamba', 'tcas', 'dger'])
    )
    log_string(f"✓ 模型参数总数:     {total_params:,}")
    log_string(f"✓ 可训练参数:       {trainable_params:,}")
    log_string(f"✓ V3新增模块参数:   {v3_module_params:,} "
               f"({v3_module_params/total_params*100:.1f}%)")
    log_string(f"✓ 模型大小:         ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    if args.progressive_training:
        frozen_count = freeze_domain_layers(model)
        log_string(f"\n渐进式训练: 前 {args.freeze_epochs} 轮冻结 {frozen_count} 个域自适应参数")
    else:
        log_string("\n标准训练: 所有参数同时训练")

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    log_string(f"✓ 优化器: {args.optimizer}")
    log_string(f"✓ 初始学习率: {args.learning_rate}")
    log_string(f"✓ 权重衰减: {args.decay_rate}")
    log_string(f"✓ 损失权重: λ1(L_elev)={args.lambda1}, λ2(L_TSCL)={args.lambda2}, "
               f"λ3(L_scene)={args.lambda3}, λ4(L_DTM)={args.lambda4}")

    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    if args.mixed_precision:
        log_string("✓ 启用混合精度训练 (AMP)")

    start_epoch = 0
    best_iou = 0
    best_rmse = float('inf')
    best_combined_score = 0
    global_epoch = 0

    if args.resume:
        if os.path.exists(args.resume):
            log_string(f"\n从checkpoint恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint.get('best_iou', 0)
            best_rmse = checkpoint.get('best_rmse', float('inf'))
            global_epoch = checkpoint.get('global_epoch', start_epoch)
            log_string(f"✓ 从epoch {start_epoch} 恢复，最佳mIoU: {best_iou:.4f}, "
                       f"最佳RMSE: {best_rmse:.4f}m")
        else:
            log_string(f"警告: checkpoint不存在: {args.resume}")

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = momentum

    training_history = {
        'train_loss': [],
        'train_l_cls': [],
        'train_l_elev': [],
        'train_l_tscl': [],
        'train_l_dtm': [],
        'train_l_scene': [],
        'train_acc': [],
        'train_rmse': [],
        'val_loss': [],
        'val_miou': [],
        'val_oa': [],
        'val_rmse': [],
        'learning_rate': [],
        'scene_natural': [],
        'scene_urban': []
    }

    log_string("\n" + "=" * 80)
    log_string("开始训练（TSRNet V3）...")
    log_string("=" * 80)

    for epoch in range(start_epoch, args.epoch):
        log_string(f'\n{"="*80}')
        log_string(f'*** Epoch {epoch + 1}/{args.epoch} (Global: {global_epoch + 1}) ***')
        log_string("="*80)

        if args.progressive_training and epoch == args.freeze_epochs:
            trainable_after = unfreeze_all_layers(model)
            if args.optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.learning_rate * 0.1,
                    weight_decay=args.decay_rate
                )
            log_string(f"\n🔓 解冻所有层！可训练参数: {trainable_after:,}")
            log_string(f"  调整学习率为: {args.learning_rate * 0.1:.6f}")

        current_lr = optimizer.param_groups[0]['lr']
        log_string(f'学习率: {current_lr:.6f}')
        training_history['learning_rate'].append(current_lr)

        momentum = max(0.1 * (0.5 ** (epoch // 20)), 0.01)
        model.apply(lambda x: bn_momentum_adjust(x, momentum))

        model.train()
        num_batches = len(trainDataLoader)
        total_correct, total_seen, loss_sum = 0, 0, 0

        l_cls_sum, l_elev_sum, l_tscl_sum, l_dtm_sum, l_scene_sum = 0, 0, 0, 0, 0
        rmse_train_list = []
        scene_natural_list = []
        scene_urban_list = []
        uncertainty_list = []
        lambda1_list = []
        lambda2_list = []

        optimizer.zero_grad()

        for i, (points, target, xyz_raw) in tqdm(
            enumerate(trainDataLoader), total=num_batches,
            smoothing=0.9, desc="Training", leave=False, ascii=True
        ):
            points = points.float().to(device)
            target = target.long().to(device)
            xyz = xyz_raw.float().to(device)
            points = points.transpose(2, 1)

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(points, gt_labels=target)
                    loss, loss_dict = criterion(outputs, target, xyz=xyz)
                    loss = loss / args.gradient_accumulation

                scaler.scale(loss).backward()

                if (i + 1) % args.gradient_accumulation == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(points, gt_labels=target)
                loss, loss_dict = criterion(outputs, target, xyz=xyz)
                loss = loss / args.gradient_accumulation
                loss.backward()

                if (i + 1) % args.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            seg_pred = outputs['output']
            pred_choice = seg_pred.cpu().data.argmax(dim=2).numpy().reshape(-1)
            batch_label = target.cpu().numpy().reshape(-1)
            total_correct += np.sum(pred_choice == batch_label)
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item() * args.gradient_accumulation

            l_cls_sum   += loss_dict.get('l_cls',   0.0)
            l_elev_sum  += loss_dict.get('l_elev',  0.0)
            l_tscl_sum  += loss_dict.get('l_tscl',  0.0)
            l_dtm_sum   += loss_dict.get('l_dtm',   0.0)
            l_scene_sum += loss_dict.get('l_scene', 0.0)
            if 'lambda1' in loss_dict:
                lambda1_list.append(loss_dict['lambda1'])
                lambda2_list.append(loss_dict['lambda2'])

            if i % 50 == 0 and 'pred_delta_z' in outputs:
                with torch.no_grad():
                    rmse = compute_rmse_from_xyz(
                        outputs['pred_delta_z'], xyz, target
                    )
                    rmse_train_list.append(rmse)

            if i % 50 == 0 and 'scene_weights' in outputs:
                sw = outputs['scene_weights'].cpu().detach().numpy()
                scene_natural_list.append(sw[:, 0].mean())
                scene_urban_list.append(sw[:, 1].mean())
                uncertainty_list.append(outputs['uncertainty'].mean().item())

        train_loss = loss_sum / num_batches
        train_acc = total_correct / float(total_seen)
        train_rmse = np.mean(rmse_train_list) if rmse_train_list else 0.0

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['train_l_cls'].append(l_cls_sum / num_batches)
        training_history['train_l_elev'].append(l_elev_sum / num_batches)
        training_history['train_l_tscl'].append(l_tscl_sum / num_batches)
        training_history['train_l_dtm'].append(l_dtm_sum / num_batches)
        training_history['train_l_scene'].append(l_scene_sum / num_batches)
        training_history['train_rmse'].append(train_rmse)

        log_string(f'\n训练结果:')
        log_string(f'  总Loss:          {train_loss:.4f}')
        log_string(f'  Accuracy:        {train_acc:.4f}')
        log_string(f'  ├─ L_cls:        {l_cls_sum/num_batches:.4f}')
        if lambda1_list:
            log_string(f'  ├─ L_elev: {l_elev_sum / num_batches:.4f}  [λ1={np.mean(lambda1_list):.3f}]')
        else:
            log_string(f'  ├─ L_elev: {l_elev_sum / num_batches:.4f}')
        log_string(f'  ├─ L_TSCL:       {l_tscl_sum/num_batches:.4f}'
                   f'  [λ2={np.mean(lambda2_list):.3f}]' if lambda2_list else f'  ├─ L_TSCL:       {l_tscl_sum/num_batches:.4f}')
        log_string(f'  ├─ L_DTM:        {l_dtm_sum/num_batches:.4f}')
        log_string(f'  └─ L_scene:      {l_scene_sum/num_batches:.4f}')
        log_string(f'  训练RMSE(采样):  {train_rmse:.4f} m')

        if scene_natural_list:
            avg_natural = np.mean(scene_natural_list)
            avg_urban = np.mean(scene_urban_list)
            training_history['scene_natural'].append(avg_natural)
            training_history['scene_urban'].append(avg_urban)
            log_string(f'  场景检测:        自然地形={avg_natural:.3f}, 城市环境={avg_urban:.3f}')
        if uncertainty_list:
            log_string(f'  平均不确定性:    {np.mean(uncertainty_list):.4f}')

        if (epoch + 1) % args.eval_freq == 0:
            log_string(f'\n---- Epoch {epoch + 1} 验证 ----')

            val_metrics = evaluate_model(
                valDataLoader, model, criterion, NUM_CLASSES,
                seg_label_to_cat, desc="Validating",
                use_amp=args.mixed_precision, device=device,
                monitor_uncertainty=True
            )

            if val_metrics:
                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_miou'].append(val_metrics['mIoU'])
                training_history['val_oa'].append(val_metrics['OA'])
                training_history['val_rmse'].append(val_metrics.get('rmse', 0.0))

                log_string(f"验证指标:")
                log_string(f"  Loss:            {val_metrics['loss']:.4f}")
                log_string(f"  OA:              {val_metrics['OA']:.4f}")
                log_string(f"  MAcc:            {val_metrics['MAcc']:.4f}")
                log_string(f"  mIoU:            {val_metrics['mIoU']:.4f}  ← 主分类指标")
                if 'rmse' in val_metrics:
                    log_string(f"  RMSE:            {val_metrics['rmse']:.4f} m ± "
                               f"{val_metrics.get('rmse_std', 0):.4f} m  ← V3新增指标")

                if 'scene_natural_mean' in val_metrics:
                    log_string(f"  场景类型:        自然地形={val_metrics['scene_natural_mean']:.3f}, "
                               f"城市环境={val_metrics['scene_urban_mean']:.3f}")
                if 'avg_uncertainty' in val_metrics:
                    log_string(f"  不确定性:        {val_metrics['avg_uncertainty']:.4f}")
                    log_string(f"  地面置信度:      {val_metrics['avg_ground_confidence']:.4f}")

                log_string(f"\n类别IoU:")
                for class_name, iou in val_metrics['class_iou'].items():
                    log_string(f"  {class_name:12s}: {iou:.4f}")

                current_mIoU = val_metrics['mIoU']
                current_rmse = val_metrics.get('rmse', float('inf'))

                if args.save_metric == 'combined':
                    rmse_normalized = min(current_rmse / 2.0, 1.0)
                    rmse_score = 1.0 - rmse_normalized
                    miou_weight = 1.0 - args.rmse_weight_in_combined
                    combined_score = miou_weight * current_mIoU + args.rmse_weight_in_combined * rmse_score
                    is_best = combined_score > best_combined_score
                    if is_best:
                        best_combined_score = combined_score
                        best_iou = current_mIoU
                        best_rmse = current_rmse
                else:
                    is_best = current_mIoU >= best_iou
                    if is_best:
                        best_iou = current_mIoU
                        best_rmse = current_rmse

                if is_best:
                    savepath = checkpoints_dir / 'best_model.pth'
                    log_string(f'\n💾 保存最佳模型:')
                    log_string(f'   mIoU = {best_iou:.4f}, RMSE = {best_rmse:.4f} m')
                    if args.save_metric == 'combined':
                        log_string(f'   联合得分 = {best_combined_score:.4f}')
                    state = {
                        'epoch': epoch,
                        'global_epoch': global_epoch,
                        'best_iou': best_iou,
                        'best_rmse': best_rmse,
                        'best_combined_score': best_combined_score,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'args': vars(args),
                        'model_version': 'V3_TSRNet',
                        'mamba_d_state': args.mamba_d_state,
                        'mamba_d_conv': args.mamba_d_conv,
                        'mamba_expand': args.mamba_expand,
                    }
                    torch.save(state, savepath)

                log_string(f'历史最佳: mIoU={best_iou:.4f}, RMSE={best_rmse:.4f} m')

        if (epoch + 1) % args.save_freq == 0:
            savepath = checkpoints_dir / f'checkpoint_epoch_{epoch+1}.pth'
            state = {
                'epoch': epoch,
                'global_epoch': global_epoch,
                'best_iou': best_iou,
                'best_rmse': best_rmse,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string(f'✓ 保存checkpoint: epoch {epoch+1}')

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        global_epoch += 1

    history_path = experiment_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    log_string(f"\n✓ 训练历史保存至: {history_path}")

    log_string('\n' + '=' * 80)
    log_string('训练完成！')
    log_string('=' * 80)

    final_savepath = checkpoints_dir / 'final_model.pth'
    state = {
        'epoch': args.epoch - 1,
        'global_epoch': global_epoch,
        'best_iou': best_iou,
        'best_rmse': best_rmse,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_version': 'V3_TSRNet',
    }
    torch.save(state, final_savepath)

    log_string(f'✓ 最终模型已保存: {final_savepath}')
    log_string(f'✓ 最佳验证 mIoU:  {best_iou:.4f}')
    log_string(f'✓ 最佳验证 RMSE:  {best_rmse:.4f} m')
    log_string('\n【V2 → V3 训练脚本改动汇总】')
    log_string('  1. DataLoader解包: (points, target) → (points, target, xyz_raw)')
    log_string('  2. model.forward:  model(points) → model(points, gt_labels=target)')
    log_string('  3. loss.forward:   criterion(outputs, target, xyz=xyz)  ✓ 直接调用forward')
    log_string('  4. 验证evaluate:   同步适配三值DataLoader，新增RMSE计算')
    log_string('  5. 损失监控:       新增L_elev/L_TSCL/L_DTM分项打印')
    log_string('  6. 最佳模型策略:   支持mIoU/combined(mIoU+RMSE)双模式')
    log_string('  7. 历史记录:       新增val_rmse/train_rmse等V3专属指标')


if __name__ == '__main__':
    args = parse_args()
    main(args)