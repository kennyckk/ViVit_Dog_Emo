from flask import Flask, render_template, request
import os
import random

from ensemble_predict import inference
from crop_videos import FCM_process

app = Flask(__name__,static_folder='static')

#FCM
CONFIG='./final_model/fcm/custom.yaml'
WEIGHT = './final_model/fcm/best.pt'

# FAM Configuration
INFERENCE_MODE="max"
class_map={0:"Negative Emotions: Sad or Angry", 1:"Positive Emotions: Happy or Relaxed"}

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        tag=random.randint(1,100)
        # Handle the uploaded video
        video = request.files['video']
        # Save the video to a desired location or perform any other processing
        global raw_path
        filename=f"raw{tag}.mp4"
        raw_path='./static/'+filename
        video.save(raw_path) #save the raw video
        # Render the template with the uploaded video
        return render_template('system.html',filename=filename)

    # If it's a GET request, render the upload form
    return render_template('video.html')

@app.route('/fam_process', methods=['POST'])
def fam_process():
    pred = inference(INFERENCE_MODE, os.path.abspath(raw_path), score=False)
    prediction = class_map[pred]

    return prediction

@app.route('/fcm', methods=['GET'])
def fcm_page():
    return render_template('system.html')

@app.route('/fcm_process', methods=['POST'])
def fcm_process():
    video_name = raw_path.split('/')[-1]  # get the name of the mp4
    video_path = os.path.abspath(os.path.join(raw_path, os.pardir))  # get abs path of the static folder
    FCM_process(video_path, video_name, CONFIG, WEIGHT)

    return render_template('system.html', display=True, fcm_vid="fcm_"+video_name)

if __name__ == '__main__':
    app.run(debug=True)