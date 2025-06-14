import os
import sys
sys.path.append('reprojection_error/droid_slam')

from reprojection_error.reprojection_error import ReprojectionErrorMetric
import glob
import tempfile
import decord
import imageio.v3 as iio

    
if __name__ == "__main__":
    # Example usage
    metric = ReprojectionErrorMetric()
    video_path = "/home/lff/data1/wdk/workspace/CVPR2026/WorldScore/debug/000000.mp4"  # or a directory

    rendered_images = []
    if os.path.isfile(video_path) and video_path.lower().endswith('.mp4'):
        with tempfile.TemporaryDirectory() as tmpdirname:
            vr = decord.VideoReader(video_path)
            for idx, frame in enumerate(vr):
                frame_path = os.path.join(tmpdirname, f"frame_{idx:06d}.png")
                iio.imwrite(frame_path, frame.asnumpy())
                rendered_images.append(frame_path)
            score = metric.compute_scores(rendered_images)
            print(f"Reprojection Error Score: {score}")
    elif os.path.isdir(video_path):
        # Accept common image extensions
        exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        for ext in exts:
            rendered_images.extend(sorted(glob.glob(os.path.join(video_path, ext))))
        score = metric.compute_scores(rendered_images)
        print(f"Reprojection Error Score: {score}")
    else:
        raise ValueError(f"video_path {video_path} is neither a valid .mp4 file nor a directory.")