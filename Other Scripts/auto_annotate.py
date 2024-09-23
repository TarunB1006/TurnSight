# code to perform auto annotation of images using the trained YOLO model
#modified from source code of yolov8

from ultralytics import YOLO
from pathlib import Path
def auto_annotate(data, det_model="yolov8x.pt", device="", output_dir=None):
    """
    Automatically annotates images using a YOLO object detection model.

    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str, optional): Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.
        device (str, optional): Device to run the models on. Defaults to an empty string (CPU or GPU, if available).
        output_dir (str | None | optional): Directory to save the annotated results.
            Defaults to a 'labels' folder in the same directory as 'data'.

    Example:
        python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data='ultralytics/assets', det_model='yolov8n.pt')
        
    """
    det_model = YOLO(det_model)
    
    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device)

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  
        if len(class_ids):
            boxes = result.boxes.xyxy  
            img_height, img_width = result.orig_img.shape[:2]

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for i in range(len(boxes)):
                    box = boxes[i].tolist()
                    class_id = class_ids[i]
                    x_center = (box[0] + box[2]) / 2 / img_width
                    y_center = (box[1] + box[3]) / 2 / img_height
                    width = (box[2] - box[0]) / img_width
                    height = (box[3] - box[1]) / img_height
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

auto_annotate(data=r'images16', det_model=r"best.pt",output_dir=r'labels16')
