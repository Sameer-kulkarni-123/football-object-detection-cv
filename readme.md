#  Football Object Detection and Tracking with Computer Vision

This project is a computer vision-based **football match analysis system** capable of detecting and tracking **players**, **referees**, and the **ball** in video.

##  Features

- **Player Detection & Tracking**
  - Players are detected with an **ellipse** underneath them.
  - Each player is assigned a **unique tracking ID** that remains constant across frames.
  - The ellipse color is matched to the **team’s color** for better distinction.

- **Referee Detection**
  - Referees are also detected and tracked.
  - They are identified by  **green**

- **Ball Tracking**
  - The ball is highlighted with a **blue triangle** above it.
  - This makes it easy to track the ball.




# To see the immediate output of the model

- The ```/model-outputs``` folder contains the final result generated by the model after processing the input video ```input-video/test2.mp4``` 


- Youtube link to the final [output_video](https://youtu.be/Os81uQ5xtdg)

# To run the model on your own input video
- ## Download test data
  You can download sample test videos from this [Kaggle Dataset](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv?resource=download)

- ## Download trained model weights
  Download the model weights from [Here](https://drive.google.com/file/d/1CNWF2tNk88RKVnxDQeNuq5fox-QsE4Kv/view?usp=sharing)

  Place the downloaded file under the ```/models``` folder

- ## Change  ```main.py``` 
  In ```main.py``` in ```
  tracks = tracker.get_obj_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stub.pkl")``` change ```read_from_stub=False ```

  This will make sure that new data is created and used rather than the cached old data

# To train your own model
- use the notebook ```/training/football_training_yolo_v5.ipynb``` to train your own model

- Its recommended to use Google Colab with a GPU

- Get the Dataset Link to train the model from [Here](https://app.roboflow.com/sam-6tcdr/football-players-detection-3zvbc-f1yjq/1) : Select the format as ```YOLO v5 PyTorch``` and Select ```Show Download Code```

- Replace the Code in the 2nd Cell with the download code

- The weights of the Trained model will the available at ```/runs/detect/train/weights/best.pt``` after the execution

- Place the weights ```best.pt``` under the ```/models``` folder


