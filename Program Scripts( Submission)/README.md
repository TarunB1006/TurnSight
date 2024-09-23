# The Bengaluru Mobility Challenge, 2024
## Team "Get Fined"

Members: Sundarakrishnan N, Sohan Varier, Tarun Bhupathi, Manaswini SK of RV College of Engineering

More detailed explainations can be found in the report: https://drive.google.com/file/d/1YZztqHRN1J5TLh3QNKnYMgrsRQ3Dhf7d/view?usp=drive_link

### Docker
A Dockerfile has been created to install necessary libraries including CUDA for the model to be able to use GPUs. The docker file can be built and run in a simple way.\
Build: `docker build -t username/imagename:version`\
Push: `docker push  username/imagename:version`\
Run: `docker run --rm --runtime=nvidia --gpus all -v 'YOUR STORAGE MOUNT':/app/data username/imagename:version python3 app.py input.json output.json`

The run command takes the input and output json file to read,process ands save the results in. Build creates the docker container and push command is used to push it to the docker repository so that anyone can pull the image and run the same.

### Scripts and Files

#### Program Scripts
This folder contains all the files needed to run the pipeline
1. **app.py** : The main driver code that has to be run. Takes an input JSON file and output JSON file as CLI arguments that provide the video paths and the path to the final output counts. 

2. **best.pt** : The most important file, our trained YOLOv8 model to detect the 7 classes of vehicles.

3. **config.py** : This file contains a dictionary of the co-ordinates of the turning pattern detection boxes required for each camera location/junction.

4. **outputTemplate.py** : Here, the output format required by the organisers is stored, which is again a dictionary of every turning pattern possible, for both counts and predictions, which is to be converted and submitted in JSON format.

5. **customCounter.py** : An *ultralytics* source code for creating a counter object that we modified based on our requirements.

6. **video_processor.py** : This file houses the *VideoProcessor* class that does the video processing to detect, track and count the turns made by the various classes of vehicles. 

7. **forecasting.py** : The *Forecaster* class is located here, which uses the count data collected while counting, to produce a prediction of the turn counts for the future. Preprocessing of the data logged onto the excel also takes place here.

8. **output_handler.py** : Just a simple script to process the derived outputs into the dictionary defined by *outputTemplate.py*.

Only *app.py* is supposed to be run, the other files cannot run on their own.\
Command to run: `python3 app.py input.json output.json`\
Format for *input.json*:
```
{
   "Cam_ID": 
    {
        "Vid_1": "/app/data/Cam_ID_vid_1.mp4",
        "Vid_2": "/app/data/Cam_ID_1_vid_2.mp4"
    }
}
```

#### Other Scripts
These are some other scripts used to ease the process of trainng and development but is not needed to run the framework.
1. **extract_images.py** : We used this script to extract images from the video downloaded from the dataset every *n* frames which we can set based on the number of images required.

2. **auto_annotate.py** : After making a basic model, we ran the extracted images through the model to annotate the images for us, and we would verify/edit the annotations. This script automated the annotation process and saved us a lot of time.

3. **data_split.py** : A simple script to split the images dataset into training, testing and validation sets.

4. **stream.py** : This code lets us view the YOLO model in action on a live video. It shows us the predictions being made in real-time in the video.

5. **capture_coordinates.py** : This script allowed us to simplify the process of creating the turn count boxes at the junctions. We simply opened the screenshot of the junction provided by the organisers and clicked on the corners of the box, and the pixel values are automatically logged.

6. **view.py** : Code that lets us view the turning boxes created against the actual images of the junction for easier interpretation. 

7. **data.yaml** : This file is used to specify dataset location during training, and holds the list of classes.

8. **predict_arima.py** : The script to test various forecasting models and methods using ARIMA, by tuning parameters, using different types of datasets etc.

9. **data_combine.py** : A simple script to combine the annotated image folders by all the team members.

### requirements.txt

#### Requirements for Program Codes

1. **opencv_python_headless** *4.10.0.84* : Used to read and extract content from the video files.

2. **ultralytics** *8.2.58* : The library that contains YOLOv8, the model we used for vehicle detection and counting.

3. **pandas** *1.5.3* : The basic data structure used throughout the project, pandas dataframes.

4. **pmdarima** *2.0.4* : Library that contains the *auto_arima* forecasting model.

5. **Shapely** *2.0.6* : A dependency of *ultrlytics'* *ObjectCounter* class.

6. **statsmodels** *0.14.2* : Library that contains various statistical forecasting and data smoothening methods that we tried. 

Run `pip3 install -r requirements.txt` in the Program Scripts folder to install all the dependencies.
#### Extra requirements for Other Scripts

1. **matplotlib** *3.7.1* : A library used to make plots and charts in python.

2. **numpy** *2.1.0* : Allows for creation and ease of manipulation on multi-dimensonal arrays in python.

3. **prophet** *1.1.5* : A forecasting procedure implemented by Facebook. We tried using this model for forecasting.

4. **scikit_learn** *1.0.2*: Open source machine learning library for python, that contains various tools for data preparation, machine learning models etc. We used its *train_test_split* class to split the data into training and testing sets.

5. **opencv_python** *4.10.0.84* : Used to read and extract content from the video files and also display them.

### Open-Source Material

**YOLOv8** by *Ultralyitcs* is an open source, real-time object detection and image segmentation model.

**labelimg** is an annotation tool that provides features to draw and edit bounding boxes in the format required by YOLOv8.

### System Requirements

- CPU - Core i5
- GPU - NVIDIA GTX 1650
- RAM - 8 GB
- SATA - 10 GB
- Around 1GB of GPU memory would be used for realtime inference.

