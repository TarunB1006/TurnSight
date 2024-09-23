# script to capture coordinates of turning boxes

import cv2

# Load the image
image_path = r"C:\Python Projects\police_hackathon\v8_data\v8_data_combined\train\images\Devasandra_Sgnl_JN_FIX_3_time_2024-05-14T07_30_02_011_frame_00095.jpg"
re_image = cv2.imread(image_path)

# Store coordinates in a list
coordinates = []
sets_of_coordinates = []

image = cv2.resize(re_image, (1440, 810))

# Function to capture mouse click events
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add coordinates to the list
        coordinates.append((x, y))
        
        # Display the coordinates on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str((x, y)), (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', image)
        
        # Check if we have captured 4 points
        if len(coordinates) == 4:
            print("Captured Coordinates:", coordinates)
            sets_of_coordinates.append(coordinates.copy())
            # Clear the list for the next set of coordinates
            coordinates.clear()

# Display the image and set mouse callback
cv2.imshow('image', image)
cv2.setMouseCallback('image', click_event)

# Wait until any key is pressed
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()

# Print all sets of coordinates
print("\nAll sets of captured coordinates:")
for i, coord_set in enumerate(sets_of_coordinates, 1):
    print(f"'J{i}': {coord_set},")