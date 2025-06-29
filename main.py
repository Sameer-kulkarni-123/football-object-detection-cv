from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
  video_frames = read_video('input-video/test2.mp4')  

  tracker = Tracker('models/best1.pt')
  tracks = tracker.get_obj_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stub.pkl")

  team_assigner = TeamAssigner()
  team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

  for frame_num, player_track in enumerate(tracks['players']):
    for player_id, track in player_track.items():
      team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)

      tracks['players'][frame_num][player_id]['team'] = team
      tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


  output_video_frames = tracker.draw_annotations(video_frames, tracks)

  save_video(output_video_frames, 'model-outputs/output-video.avi')


if __name__ == '__main__':
  main()