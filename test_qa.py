from data_qa.vqa import get_video_quality

score = get_video_quality("/app/data/raw_data/1.0.0/exterior_image_1.mp4", device="cuda")

print(score)