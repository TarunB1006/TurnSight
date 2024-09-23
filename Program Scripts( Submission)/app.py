import argparse
import json
from video_processor import VideoProcessor
from forecasting import Forecaster
from config import locations_config as config
from output_handler import OutputHandler

def main(json_input, output_file):
    try:
        with open(json_input, 'r') as file:
            data = json.load(file)  # Load the JSON input
    except:
        return "Error reading JSON file"

    results = {}
    video_processor = VideoProcessor()
    output_handler = OutputHandler()
    forecaster = Forecaster()

    for cam_id, videos in data.items():
        print(f"Processing camera: {cam_id}")
        location_config = config[cam_id]

        # Step 1: Counting vehicles
        cumulative_counts, frame_count_dict = video_processor.process_videos_and_compile_counts(cam_id, videos, location_config)

        # Step 2: Forecasting vehicle counts
        predicted_counts = forecaster.forecast_vehicle_counts(cumulative_counts, frame_count_dict, video_processor.fps)

        results[cam_id] = {
            'Cumulative Counts': cumulative_counts,
            'Predicted Counts': predicted_counts
        }
        results = output_handler.update_counts_with_camera_id(results, cam_id)

    # Save the final result as JSON to the specified output file
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video segments from JSON input.')
    parser.add_argument('json_input', type=str, help='Enter JSON input')
    parser.add_argument('output_file', type=str, help='Enter filename to save output')

    args = parser.parse_args()

    main(args.json_input, args.output_file)
