import torch
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator

from run_fn.test import test_fn
from models.tracker import PerformanceTracker
from configs.algo_config import SDGFAConfig
from utils.loss_fn import emb_loss_fn, mps_loss_fn
from utils.misc import set_grad_flag


def sdgfa_train_fn(model, train_dataset, val_dataset, gt_stylef_dist, algo_config: SDGFAConfig, run=None):
    """
    Train the model using Accelerate for efficient multi-device support.
    """
    # Initialize Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=algo_config.grad_accum_steps)

    # Prepare model, optimizer, and data loader for Accelerate
    model = model.to(accelerator.device)
    if algo_config.face_model == "facenet-vggface2":
        face_model = InceptionResnetV1(pretrained='vggface2').eval()
        set_grad_flag(face_model, False)
        face_model = face_model.to(accelerator.device)
    else:
        raise ValueError(f"Invalid face model {algo_config.face_model}")

    train_loader = DataLoader(train_dataset, batch_size=algo_config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.control_module.parameters(), lr=algo_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=algo_config.lr_decay_step, gamma=algo_config.lr_decay_weight)
    tracker = PerformanceTracker(early_stop_epochs=algo_config.early_stop_epochs)

    # Prepare for distributed training
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for i_epoch in range(algo_config.num_epochs):
        if i_epoch >= algo_config.warmup_epochs:
            scheduler.step()

        _sdgfa_train_loop(model, face_model, train_loader, optimizer,
                          gt_stylef_dist, algo_config.mps_lower_bound,
                          run=run, accelerator=accelerator, i_epoch=i_epoch)

        # Validation
        metric_dict = test_fn(model, face_model=face_model, test_dataset=val_dataset, gt_stylef_dist=gt_stylef_dist, algo_config=algo_config)
        if run is not None:
            _metric_dict = {f"val/{key}": value for key, value in metric_dict.items()}
            _metric_dict["lr"] = scheduler.get_last_lr()[0]
            run.log(_metric_dict, step=i_epoch)
        model_state_dict = model.state_dict()
        early_stop_flag = tracker.update(metric_dict, model_state_dict)
        if early_stop_flag:
            break

    best_state_dict = tracker.export_best_model_state_dict()
    model.load_state_dict(best_state_dict)
    best_metric_dict = tracker.export_best_metric_dict()

    if run is not None:
        Path(algo_config.log_dir).mkdir(parents=True, exist_ok=True)
        torch.save(best_state_dict, f"{algo_config.log_dir}/sdgfa.pth")

    return best_metric_dict


def _sdgfa_train_loop(model, face_model, train_loader, optimizer,
                      gt_style_dist, mps_lower_bound, run, accelerator, i_epoch):
    """
    Main training loop using Accelerate.
    """
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"SD-GFA Training, Epoch {i_epoch}")
    optimizer.zero_grad()

    for i_batch, (img_x1, img_x2, img_y, key1, key2) in enumerate(train_loader):
        img_x1, img_x2, img_y, key1, key2 = (
            img_x1.to(accelerator.device),
            img_x2.to(accelerator.device),
            img_y.to(accelerator.device),
            key1.to(accelerator.device),
            key2.to(accelerator.device),
        )

        with accelerator.accumulate(model):
            out_x1_k1, info_x1_k1 = model.forward_with_hidden(img_x1, key1)
            out_x1_k2, info_x1_k2 = model.forward_with_hidden(img_x1, key2)
            out_x2_k1, info_x2_k1 = model.forward_with_hidden(img_x2, key1)
            out_y_k1, info_y_k1 = model.forward_with_hidden(img_y, key1)

            stylef_x1_k1 = info_x1_k1['style_latents']
            stylef_x1_k2 = info_x1_k2['style_latents']
            stylef_x2_k1 = info_x2_k1['style_latents']
            stylef_y_k1 = info_y_k1['style_latents']
            stylef = torch.cat([stylef_x1_k1, stylef_x1_k2, stylef_x2_k1, stylef_y_k1], dim=0)

            f_out_x1 = face_model(img_x1)
            f_out_x1_k1 = face_model(out_x1_k1)
            f_out_x1_k2 = face_model(out_x1_k2)
            f_out_x2_k1 = face_model(out_x2_k1)
            f_out_y_k1 = face_model(out_y_k1)
            del out_x1_k1, out_x1_k2, out_x2_k1, out_y_k1, img_x1, img_x2, img_y, key1, key2

            # Loss calculation
            pri_loss = emb_loss_fn(f_out_x1, f_out_x1_k1, 0)
            con_loss = emb_loss_fn(f_out_x1_k1, f_out_x1_k2, 0)
            intra_loss = emb_loss_fn(f_out_x1_k1, f_out_x2_k1, 1)
            inter_loss = emb_loss_fn(f_out_x1_k1, f_out_y_k1, 0)
            mps_loss = mps_loss_fn(stylef, gt_style_dist, mps_lower_bound)

            loss = 0.1 * pri_loss + con_loss + intra_loss + inter_loss + 20 * mps_loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        loss_dict = {'loss': loss.item(), 'pri_loss': pri_loss.item(), 'con_loss': con_loss.item(),
                     'intra_loss': intra_loss.item(), 'inter_loss': inter_loss.item(), 'mps_loss': mps_loss.item()}
        if run is not None:
            batch_step = i_batch + len(train_loader) * i_epoch
            _loss_dict = {f"train/{key}": value for key, value in loss_dict.items()}
            run.log(_loss_dict, step=batch_step)
        pbar.set_postfix(loss_dict)
        pbar.update(1)
