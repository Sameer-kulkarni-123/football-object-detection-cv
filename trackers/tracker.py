from ultralytics import YOLO
import supervision as sv
import os
import pickle

class Tracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.tracker = sv.ByteTrack()

  def detect_objects_in_frame(self, frames, batch_size=20):
    
    detections = []

    for i in range(0, len(frames), batch_size):
      batch_detections = self.model.predict(frames[i:i+batch_size], conf=0.1)
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
