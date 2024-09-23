# This file houses the VideoProcessor class that does the video processing to detect, track and count the turns made by the various classes of vehicles. 

import cv2
from collections import Counter
from ultralytics import YOLO
from customCounter import ObjectCounter

class VideoProcessor:
    def __init__(self):
        self.model = YOLO(r"best.pt")
        self.fps = 0
        self.vehicle_classes = ['Car', 'Bus', 'Truck', 'Three Wheeler', 'Two Wheeler', 'LCV', 'Bicycle']

    # initialize the counters for each junction
    def initialize_counters(self, location_config):
        counters = {}
        for junction, points in location_config['regions'].items():
            counters[junction] = ObjectCounter(
                view_img=False,
                reg_pts=points,
                names=self.model.names,
                draw_tracks=True,
                line_thickness=2
            )
        return counters

    # process the video frame by frame
    def process_video(self, location_config, video_path, frame_count):
        new_width = 1440
        new_height = 810
        counters = self.initialize_counters(location_config)
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Error reading video file: {video_path}"
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video frame rate (FPS): {self.fps}")

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video processing completed.")
                break

            im0_resized = cv2.resize(im0, (new_width, new_height))
            tracks = self.model.track(im0_resized, persist=True, show=False, conf=0.5)
            frame_count += 1

            for junction, counter in counters.items():
                im0_resized = counter.start_counting(im0_resized, tracks, frame_count)

        cap.release()
        return counters, frame_count

    # counting the turning patterns
    def count_turning_patterns(self, counters, turning_patterns, frame_count_dict, offset):
        count_dict = {pattern: {cls: 0 for cls in self.vehicle_classes} for pattern in turning_patterns}
        for pattern, (src, dst) in turning_patterns.items():
            for count_id in counters[src].count_ids.keys():
                if counters[src].count_ids[count_id][1] == 'source':
                    if count_id in counters[dst].count_ids and counters[dst].count_ids[count_id][1] == 'dest':
                        vehicle_class_str = self.vehicle_classes[int(counters[dst].count_ids[count_id][0])]
                        count_dict[pattern][vehicle_class_str] += 1
                        frame_number = counters[dst].count_ids[count_id][2]
                        if pattern not in offset:
                            frame_count_dict[pattern].append((frame_number, count_dict[pattern].copy()))
                        else:
                            frame_count_dict[pattern].append(
                                (frame_number, dict(Counter(count_dict[pattern].copy()) + Counter(offset[pattern].copy())))
                            )
        return count_dict, frame_count_dict

    # process the videos and compile the counts
    def process_videos_and_compile_counts(self, cam_id, videos, location_config):
        cumulative_counts = {}
        count_dict = {}
        frame_count_dict = {pattern: [] for pattern in location_config['turning_patterns']}
        frame = 0
        for count, (video_name, video_path) in enumerate(videos.items(), start=1):
            print(f"Processing video: {video_name} at {video_path}")
            if count == 1:
                counters, frame = self.process_video(location_config, video_path, 0)
            else:
                counters, frame = self.process_video(location_config, video_path, frame)

            count_dict, frame_count_dict = self.count_turning_patterns(counters, location_config['turning_patterns'], frame_count_dict, count_dict)

            for pattern, counts in count_dict.items():
                if pattern not in cumulative_counts:
                    cumulative_counts[pattern] = counts.copy()
                else:
                    for cls in self.vehicle_classes:
                        cumulative_counts[pattern][cls] += counts[cls]

        return cumulative_counts, frame_count_dict
