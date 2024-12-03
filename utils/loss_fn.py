import torch
import torch.nn.functional as F


def emb_loss_fn(f1, f2, l: int):
    """
    Compute the loss between two embeddings based on cosine similarity.
    params:
        f1: tensor, the first embedding
        f2: tensor, the second embedding
        l: int, l=1 for same identity, l=0 for different identity
    """
    if l ==  1:
        loss = 1 - torch.mean(F.cosine_similarity(f1, f2)) # encourage similarity to be 1
    elif l == 0:
        loss = torch.max(torch.tensor(0.), torch.mean(F.cosine_similarity(f1, f2)))
    else:
        raise ValueError(f"Invalid l value {l}")
    return loss


def mps_loss_fn(style_f, gt_stylef_dist: torch.distributions.Normal, lower_bound: float):
    """
    Compute the MPS loss between the style feature and the ground truth style feature distribution.
    params:
        style_f: tensor, the style feature
        gt_stylef_dist: torch.distributions.Normal, the ground truth style feature distribution
        lower_bound: float, the lower bound for the MPS loss
    """
    device = style_f.device
    logprob_f = gt_stylef_dist.log_prob(style_f)
    gt_stylef_mean = gt_stylef_dist.mean.to(device)
    logprob_gt = gt_stylef_dist.log_prob(gt_stylef_mean)
    log_p_lower_bound = torch.log(torch.tensor([lower_bound], device=device))

    if torch.mean(logprob_f) < log_p_lower_bound:
        mps_loss = 20*torch.mean((style_f - gt_stylef_mean)**2)
    else:
        mps_loss = torch.mean(logprob_gt-logprob_f)
    return mps_loss


def test_emb_loss_batch():
    f1s = torch.randn((2,3,128,128))
    f2s = torch.randn((2,3,128,128))
    l = 0
    print(emb_loss_fn(f1s, f2s, l))


def test_mps_loss_fn():
    style_f = torch.randn((512))
    gt_stylef = torch.randn((100, 512))
    gt_stylef_dist = torch.distributions.Normal(gt_stylef.mean(), gt_stylef.std())
    lower_bound = 0.1
    print(mps_loss_fn(style_f, gt_stylef_dist, lower_bound))


if __name__ == "__main__":
     test_mps_loss_fn()
