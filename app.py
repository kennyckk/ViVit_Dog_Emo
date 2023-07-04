from flask import Flask, render_template, request
import os

from ensemble_predict import inference


app = Flask(__name__,static_folder='static')


INFERENCE_MODE="max"
class_map={0:"Negative Emotions: Sad or Angry", 1:"Positive Emotions: Happy or Relaxed"}

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':

        # Handle the uploaded video
        video = request.files['video']
        # Save the video to a desired location or perform any other processing
        global raw_path
        raw_path='./static/'+"raw1.mp4"
        video.save(raw_path) #save the raw video
        # Render the template with the uploaded video
        return render_template('video.html',display=True)

    # If it's a GET request, render the upload form
    return render_template('video.html')

@app.route('/process', methods=['POST'])
def fam_process():
    pred = inference(INFERENCE_MODE, os.path.abspath(raw_path), score=False)
    prediction = class_map[pred]

    return prediction

if __name__ == '__main__':
    app.run(debug=True)