#!/bin/bash

VIDEO_RESULTS_ROOT="videos"
METRICS_ROOT="metrics"

# Loop through each subdirectory in VIDEO_RESULTS_ROOT
for video_dir1 in "$VIDEO_RESULTS_ROOT"/*/; do
    # Skip the 'metrics' directory
    if [ -d "$video_dir1" ] && [ "$(basename "$video_dir1")" != "metrics" ]; then
        # Construct the output file name based on the video directory name
        fvd_output_file="$METRICS_ROOT/$(basename "$video_dir1").json"
        echo $fvd_output_file
        # Run the python command for each video directory
        python common_metrics.py --video_dir2 $JSONL_PATH --video_length 15 --channel 3 --size "(224,384)" \
                --video_dir1 "$video_dir1" --output-file "$fvd_output_file"
    fi
done