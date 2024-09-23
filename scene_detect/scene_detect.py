from scenedetect import detect, ContentDetector, split_video_ffmpeg


vedioname = r"xx.mp4"
scene_list = detect(vedioname, ContentDetector())
split_video_ffmpeg(vedioname, scene_list,output_dir = '../datasets/scenes_detect')