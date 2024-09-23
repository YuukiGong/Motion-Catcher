import os
import cv2


def split_video(frame_lists, output_folder):
    folder_count = len(frame_lists)//14
    for i in range(folder_count):
        first_img = i*14
        idx = first_img
        dir_name = os.path.join(output_subfolder, f"{i}")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            for frame in frame_lists[first_img:first_img+14]:
                cv2.imwrite(os.path.join(dir_name, f"frame{idx}.jpg"), frame)
                idx = idx + 1

video_folder = "../datasets/scenes_detect"
output_folder = "../datasets/around"

# video_files = os.listdir(video_folder)[:3]


for root, dirs, files in os.walk(video_folder):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            
            
            # video_path = os.path.join(video_folder, video_file)
            output_subfolder = os.path.join(output_folder, file.split(".")[0])
            print(output_subfolder)

            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(frame_rate / 7)

            frame_count = 0
            frame_idx = 0
            frame_lists = []
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_lists.append(frame)
                    # cv2.imwrite(os.path.join(output_subfolder, f"frame{frame_idx}.jpg"), frame)

                frame_count += 1
                frame_idx += 1

            cap.release()

            split_video(frame_lists, output_subfolder)