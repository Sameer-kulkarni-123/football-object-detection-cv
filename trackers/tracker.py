from ultralytics import YOLO
import supervision as sv
import os
import pickle
import sys
import cv2
import numpy as np
sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox

class Tracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.tracker = sv.ByteTrack()

  def detect_objects_in_frame(self, frames, batch_size=20):
    
    detections = []

    for i in range(0, len(frames), batch_size):
      batch_detections = self.model.predict(frames[i:i+batch_size], conf=0.1 )
      detections += batch_detections
      # break #temporary
    
    return detections

  def get_obj_tracks(self, frames, read_from_stub=False, stub_path=None):

    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
      with open(stub_path, "rb") as f:
        tracks = pickle.load(f)
      return tracks

    tracks = {
      "players" : [], # "players" : [{"1" : {"bbox" : [...]}, "2" : {"bbox" : [... -> bbox co-ordinates]} -> objs }, {}, {} -> frames]
      "referees" : [],
      "ball" : []
    }


    detections = self.detect_objects_in_frame(frames)

    for frame_num, detection in enumerate(detections):
      cls_names = detection.names
      cls_names_inv = {v:k for k, v in cls_names.items()}
      # print(cls_names_inv)

      detection_supervision = sv.Detections.from_ultralytics(detection)
      
      for obj_id, class_id in enumerate(detection_supervision.class_id):
        if cls_names[class_id] == 'goalkeeper':
          detection_supervision.class_id[obj_id] = cls_names_inv['player']

      detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

      tracks["players"].append({})
      tracks["referees"].append({})
      tracks["ball"].append({})




      for each_obj in detection_with_tracks:
        bbox = each_obj[0].tolist()
        cls_id = each_obj[3]
        track_id = each_obj[4]

        if cls_id == cls_names_inv["player"]:
          tracks["players"][frame_num][track_id] = {"bbox":bbox}

        if cls_id == cls_names_inv["referee"]:
          tracks["referees"][frame_num][track_id] = {"bbox":bbox}

      for each_obj in detection_supervision:
        bbox = each_obj[0].tolist()
        cls_id = each_obj[3]

        if cls_id == cls_names_inv["ball"]:
          tracks["ball"][frame_num][1] = {"bbox" : bbox}

    if stub_path is not None:
      with open(stub_path, "wb") as f:
        pickle.dump(tracks, f)
      
    return tracks
      # print(detection_with_tracks)
      # break #temporary

  def draw_ellipse(self, frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, y_center = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    cv2.ellipse(frame,
                center=(x_center, y2),
                axes=(int(width), int(0.35*width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4)

    rect_height = 20
    rect_width = 40
    x1_rect = x_center - rect_width//2
    x2_rect = x_center + rect_width//2
    y1_rect = (y2 - rect_height//2) + 15
    y2_rect = (y2 + rect_height//2) + 15

    if track_id is not None:
      cv2.rectangle(frame, 
                    (int(x1_rect), int(y1_rect)),
                    (int(x2_rect), int(y2_rect)),
                    color,
                    cv2.FILLED)

      x1_text = int(x1_rect + 12)
      if track_id > 99:
        x1_text -= 10
      y1_text = int(y1_rect + 15)
      
      cv2.putText(frame, 
                  f"{track_id}",
                  (x1_text, y1_text),
                  cv2.FONT_HERSHEY_COMPLEX,
                  0.6,
                  (0,0,0),
                  2
                  )
                
    return frame
  
  def draw_triangle(self, bbox, frame, color):
    y = bbox[1]
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
      [x, y],
      [x-10, y-20],
      [x+10, y-20]
    ], dtype=np.int32).reshape((-1, 1, 2))
    cv2.drawContours(frame,
                     [triangle_points],
                     0,
                     color,
                     cv2.FILLED
                    )
    cv2.drawContours(frame,
                     [triangle_points],
                     0,
                     (0,0,0),
                     2
                    )

    return frame

  def draw_annotations(self, video_frames, tracks):
    output_frames = []

    for frame_num, frame in enumerate(video_frames):
      frame = frame.copy()

      player_dict = tracks["players"][frame_num]
      referee_dict = tracks["referees"][frame_num]
      ball_dict = tracks["ball"][frame_num]

      #Drawing Players
      for track_id, player in player_dict.items():
        color = player.get('team_color', (0, 0, 255))
        frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

      #Drawing Referees
      for track_id, referee in referee_dict.items():
        frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 0))

      #Drawing Ball
      for track_id, ball, in ball_dict.items():
        frame = self.draw_triangle(ball["bbox"], frame, (255, 0, 0))

      output_frames.append(frame)

    return output_frames