from outputTemplate import template

class OutputHandler:
    def update_counts_with_camera_id(self, partial_dict, new_camera_id):
        large_dict_template = template
        large_dict = {new_camera_id: large_dict_template["Cam_ID"]}
        
        if new_camera_id in large_dict:
            cumulative_counts = large_dict[new_camera_id]["Cumulative Counts"]
            predictive_counts = large_dict[new_camera_id]["Predicted Counts"]
            for turn, updates in partial_dict[new_camera_id]["Cumulative Counts"].items():
                if turn not in cumulative_counts:
                    cumulative_counts[turn] = updates
                else:
                    for vehicle_class, count in updates.items():
                        cumulative_counts[turn][vehicle_class] += count

            for turn, updates in partial_dict[new_camera_id]["Predicted Counts"].items():
                if turn not in predictive_counts:
                    predictive_counts[turn] = updates
                else:
                    for vehicle_class, count in updates.items():
                        predictive_counts[turn][vehicle_class] += count

        return large_dict
