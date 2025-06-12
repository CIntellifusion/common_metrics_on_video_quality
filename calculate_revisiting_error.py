import os
import sys
import argparse
import imageio
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics
# load metrics 
from calculate_psnr import img_psnr
from calculate_ssim import calculate_ssim_function 
from calculate_lpips import loss_fn 

def load_dfot_format(video_folder):
    """
    Load DFoT format videos from a folder containing GIF files.
    Returns:
        gt_tensors: torch.Tensor of shape [N, C, H/2, W]
        pred_tensors: torch.Tensor of shape [N, C, H/2, W]
    """
    to_tensor = transforms.ToTensor()
    gif_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.gif')])
    
    gt_list = []
    pred_list = []

    for gif_name in gif_files:
        gif_path = os.path.join(video_folder, gif_name)
        reader = imageio.mimread(gif_path)  # list of frames as numpy arrays
        
        if len(reader) < 2:
            print(f"[Warning] {gif_name} has less than 2 frames, skipping.")
            continue

        # Convert to PIL Images
        frames = [Image.fromarray(frame) for frame in reader]

        # Get first and last frames
        first_frame = frames[0]
        last_frame = frames[-1]

        # Convert to tensor
        first_tensor = to_tensor(first_frame)  # [C, H, W]
        last_tensor = to_tensor(last_frame)

        # Split height in half
        C, H, W = first_tensor.shape
        w_half = W // 2
        gt = first_tensor[:, :, :w_half]
        pred = last_tensor[:, :, :w_half]

        gt_list.append(gt)
        pred_list.append(pred)

        print(f"[{gif_name}] First frame shape: {first_tensor.shape}, GT: {gt.shape}, Pred: {pred.shape}")

    if not gt_list:
        raise RuntimeError("No valid GIFs found in the folder.")

    # Stack into tensors
    gt_tensors = torch.stack(gt_list)  # [N, C, H/2, W]
    pred_tensors = torch.stack(pred_list)
    print(f"GT Tensors shape: {gt_tensors.shape}, Pred Tensors shape: {pred_tensors.shape}")
    return gt_tensors, pred_tensors

def calculate_image_lpips(gt, pred, device, loss_fn):
    """
    Calculate LPIPS distance between two batches of images.

    Args:
        gt:   torch.Tensor [B, 3, H, W], value in [0, 1]
        pred: torch.Tensor [B, 3, H, W], value in [0, 1]
        device: torch device
        loss_fn: lpips.LPIPS instance

    Returns:
        dict with keys "value", "value_std"
    """
    assert gt.shape == pred.shape, "GT and Pred must have same shape"
    
    loss_fn.to(device)
    loss_fn.eval()
    
    lpips_scores = []

    # Normalize to [-1, 1]
    gt = gt * 2 - 1
    pred = pred * 2 - 1

    for i in tqdm(range(gt.shape[0]), desc="Calculating LPIPS", disable=True):
        img1 = gt[i].unsqueeze(0).to(device)   # [1, 3, H, W]
        img2 = pred[i].unsqueeze(0).to(device)

        with torch.no_grad():
            dist = loss_fn(img1, img2).mean().item()
        lpips_scores.append(dist)

    lpips_array = np.array(lpips_scores)
    return {
        "value": float(np.mean(lpips_array)),
        "value_std": float(np.std(lpips_array))
    }

def calculate_image_psnr(gt, pred):
    """
    Args:
        gt:   torch.Tensor of shape [B, 3, H, W], range [0,1]
        pred: torch.Tensor of shape [B, 3, H, W], range [0,1]

    Returns:
        dict with keys: "value", "value_std"
    """
    assert gt.shape == pred.shape, "GT and Pred must have same shape"
    
    psnr_list = []
    
    for i in tqdm(range(gt.shape[0]), desc="Calculating PSNR", disable=True):
        gt_img = gt[i].cpu().numpy()
        pred_img = pred[i].cpu().numpy()
        
        # Convert to numpy in [H, W, C] for better readability
        gt_img = np.transpose(gt_img, (1, 2, 0))  # [H, W, C]
        pred_img = np.transpose(pred_img, (1, 2, 0))
        
        psnr_val = img_psnr(gt_img, pred_img)
        psnr_list.append(psnr_val)

    psnr_array = np.array(psnr_list)
    return {
        "value": float(np.mean(psnr_array)),
        "value_std": float(np.std(psnr_array))
    }

def calculate_image_ssim(gt, pred):
    """
    Args:
        gt:   torch.Tensor of shape [B, 3, H, W], range [0,1]
        pred: torch.Tensor of shape [B, 3, H, W], range [0,1]
    Returns:
        dict with keys "value", "value_std"
    """
    assert gt.shape == pred.shape, "Input shape mismatch"
    assert gt.shape[1] == 3, "SSIM expects 3-channel images"

    ssim_scores = []

    for i in tqdm(range(gt.shape[0]), desc="Calculating SSIM", disable=True):
        img1 = gt[i].cpu().numpy()
        img2 = pred[i].cpu().numpy()
        score = calculate_ssim_function(img1, img2)
        ssim_scores.append(score)

    ssim_scores = np.array(ssim_scores)
    return {
        "value": float(np.mean(ssim_scores)),
        "value_std": float(np.std(ssim_scores))
    }

def calculate_image_fid(gt, pred, save_dir='./fid_temp'):
    """
    Returns:
        dict with 'frechet_inception_distance'
    """
    assert gt.shape == pred.shape and gt.shape[1] == 3, "Input must be [B, 3, H, W]"
    
    gt_dir = os.path.join(save_dir, "gt")
    pred_dir = os.path.join(save_dir, "pred")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    print("Saving temporary images for FID...")

    for i in tqdm(range(gt.shape[0]), desc="Saving GT/Pred images", disable=True):
        save_image(gt[i], os.path.join(gt_dir, f"{i:05d}.png"))
        save_image(pred[i], os.path.join(pred_dir, f"{i:05d}.png"))

    print("Calculating FID...")
    metrics = calculate_metrics(input1=gt_dir, input2=pred_dir,
                                cuda=torch.cuda.is_available(),
                                isc=False,
                                fid=True,
                                kid=False)

    # Optional: 清除暫存資料夾
    import shutil; shutil.rmtree(save_dir)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Load DFoT GIF folder and extract GT/Pred frames.")
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to the folder containing GIF files.")
    args = parser.parse_args()

    video_folder = args.folder
    if not os.path.isdir(video_folder):
        print(f"Error: folder '{video_folder}' does not exist.")
        sys.exit(1)

    print(f"Loading videos from: {video_folder}")
    gt, pred = load_dfot_format(video_folder)

    print(f"Loaded {gt.shape[0]} videos.")
    print(f"GT tensor shape: {gt.shape}")
    print(f"Pred tensor shape: {pred.shape}")
    print(f"GT tensor range : {gt.min().item()} to {gt.max().item()}")
    print(f"Pred tensor range : {pred.min().item()} to {pred.max().item()}")
    # save to images 
    save_dir="./cache_dir"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(gt.shape[0]):
        combined_tensor = torch.cat([gt[i], pred[i]], dim=-1)  # [3, H, 2W]
        combined_image = transforms.ToPILImage()(combined_tensor.cpu()).convert("RGB")
        combined_image.save(os.path.join(save_dir, f"gt_pred_{i:04d}.png"))
    print(f"Saved GT and Pred images to {save_dir}")
    print("Done!")
    revisit_rfid = calculate_image_fid(gt, pred, save_dir="cache_fid")
    print(f"Revisiting FID: {revisit_rfid['frechet_inception_distance']:.4f}")
    revisiting_psnr = calculate_image_psnr(gt, pred)
    print(f"Revisiting PSNR: {revisiting_psnr['value']:.2f} ± {revisiting_psnr['value_std']:.2f}")
    revisiting_lpips = calculate_image_lpips(gt, pred, device="cuda" if torch.cuda.is_available() else "cpu", loss_fn=loss_fn)
    print(f"Revisiting LPIPS: {revisiting_lpips['value']:.4f} ± {revisiting_lpips['value_std']:.4f}")
    revisiting_ssim = calculate_image_ssim(gt, pred)
    print(f"Revisiting SSIM: {revisiting_ssim['value']:.4f} ± {revisiting_ssim['value_std']:.4f}")
if __name__ == "__main__":
    main()