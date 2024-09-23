# Code that lets us view the turning boxes created against the actual images of the junction for easier interpretation.

import cv2
import numpy as np

# Function to draw bounding boxes
def draw_bounding_boxes(img, regions):
    for region_name, coord_set in regions.items():
        # Convert list to a NumPy array of points
        pts = np.array(coord_set, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # Draw a closed polygon (bounding box)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Optionally, you can display the region name on the image
        # Position the text near the top-left corner of the bounding box
        text_position = (coord_set[0][0], coord_set[0][1] - 10)
        cv2.putText(img, region_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with the bounding boxes
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the image
image_path = r"C:\Users\al1\Desktop\2.png"
image = cv2.imread(image_path)

# Resize the image if needed
image = cv2.resize(image, (1440, 810))

# Example dictionary of regions with coordinates
regions= {    
    'J1':[(14, 340), (30, 781), (1417, 786), (1425, 340)], #J1 A
    'J2':[(9, 10), (17, 333), (1426, 333), (1416, 10)], #J2 B
    }

# Draw the bounding boxes on the image
draw_bounding_boxes(image, regions)