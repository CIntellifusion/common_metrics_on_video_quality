import os
import sys
sys.path.append('reprojection_error/droid_slam')

from reprojection_error.reprojection_error import ReprojectionErrorMetric
import glob
import tempfile


if __name__ == "__main__":
    # Example usage
    metric = ReprojectionErrorMetric()
    video_path = "assets/000003.mp4"  # or a directory
    reconstruction_path = (
        "reconstructions/000000_recon.pth"  # Path to save reconstruction output, if None it won't save
    )

    rendered_images = []
    if os.path.isfile(video_path) and video_path.lower().endswith('.mp4'):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Use ffmpeg to extract frames
            # -vsync 0 to avoid duplicate frames, -q:v 2 for high quality
            os.system(
                f"ffmpeg -i '{video_path}' -vsync 0 -q:v 2 '{tmpdirname}/frame_%06d.png' -hide_banner -loglevel error"
            )
            rendered_images = sorted(glob.glob(os.path.join(tmpdirname, "frame_*.png")))
            score = metric.compute_scores(rendered_images, reconstruction_path=reconstruction_path)
            print(f"Reprojection Error Score: {score}")
    elif os.path.isdir(video_path):
        # Accept common image extensions
        exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        for ext in exts:
            rendered_images.extend(sorted(glob.glob(os.path.join(video_path, ext))))
        score = metric.compute_scores(rendered_images, reconstruction_path=reconstruction_path)
        print(f"Reprojection Error Score: {score}")
    else:
        raise ValueError(f"video_path {video_path} is neither a valid .mp4 file nor a directory.")
