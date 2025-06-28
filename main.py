from utils import read_video, save_video
from trackers import Tracker


def main():
  video_frames = read_video('input-video/test1.mp4')  
  tracker = Tracker('models/best1.pt')

  tracker.get_obj_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stub.pkl")

  # save_video(output_frames, 'model-outputs/output-video.avi')


if __name__ == '__main__':
  main()