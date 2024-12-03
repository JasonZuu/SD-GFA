import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.loss_fn import emb_loss_fn, mps_loss_fn


@torch.no_grad()
def test_fn(model, face_model, test_dataset, gt_stylef_dist, algo_config):
    """
    Evaluate the model on the test_dataset.
    
    Args:
        model: The StyleDiverseGFA model instance.
        val_dataset: The validation dataset.
        algo_config: Configuration object with training parameters.

    Returns:
        Average loss on the validation dataset.
    """
    model.eval()
    device = algo_config.device
    test_loader = DataLoader(test_dataset, batch_size=algo_config.batch_size, shuffle=False)

    pbar = tqdm(test_loader, total=len(test_loader), desc=f"SD-GFA Testing Loop")
    total_stylef = []
    total_f_out_x1 = []
    total_f_out_x1_k1 = []
    total_f_out_x1_k2 = []
    total_f_out_x2_k1 = []
    total_f_out_y_k1 = []

    for img_x1, img_x2, img_y, key1, key2 in test_loader:
        img_x1, img_x2, img_y, key1, key2 = (
            img_x1.to(device),
            img_x2.to(device),
            img_y.to(device),
            key1.to(device),
            key2.to(device),
        )

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

        total_stylef.append(stylef)
        total_f_out_x1.append(f_out_x1)
        total_f_out_x1_k1.append(f_out_x1_k1)
        total_f_out_x1_k2.append(f_out_x1_k2)
        total_f_out_x2_k1.append(f_out_x2_k1)
        total_f_out_y_k1.append(f_out_y_k1)

        pbar.update(1)
    
    total_stylef = torch.cat(total_stylef, dim=0)
    total_f_out_x1 = torch.cat(total_f_out_x1, dim=0)
    total_f_out_x1_k1 = torch.cat(total_f_out_x1_k1, dim=0)
    total_f_out_x1_k2 = torch.cat(total_f_out_x1_k2, dim=0)
    total_f_out_x2_k1 = torch.cat(total_f_out_x2_k1, dim=0)
    total_f_out_y_k1 = torch.cat(total_f_out_y_k1, dim=0)

    # Loss calculation
    pri_loss = emb_loss_fn(total_f_out_x1, total_f_out_x1_k1, 0)
    con_loss = emb_loss_fn(total_f_out_x1_k1, total_f_out_x1_k2, 0)
    intra_loss = emb_loss_fn(total_f_out_x1_k1, total_f_out_x2_k1, 1)
    inter_loss = emb_loss_fn(total_f_out_x1_k1, total_f_out_y_k1, 0)
    mps_loss = mps_loss_fn(total_stylef, gt_stylef_dist, algo_config.mps_lower_bound)
    loss = 0.1*pri_loss + con_loss + intra_loss + inter_loss + 20*mps_loss

    metric_dict = {'loss':loss, 'pri_loss': pri_loss, 'con_loss':con_loss,\
                   'intra_loss':intra_loss, 'inter_loss':inter_loss, 'mps_loss':mps_loss, }
    return metric_dict
