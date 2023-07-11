import os
from tqdm import tqdm
import cv2
import shutil
import re
import sys

from yolov5.detect import run
    
#function to crop the face of dog from videos
def crop_videos(video_path,
     video_name,
     weight="./best.pt",
     data='./data/custom.yaml',
     confidence_thres=0.6,
     yolo_dim=(640,640),
     target_folder='./cropped_vids'):

    video= os.path.join(video_path,video_name) #i.e. xxx.mp4
    #to run the detect model per video
    run(weights=weight,  # model path or triton URL
        source=video,  # file/dir/URL/glob/screen/0(webcam)
        data=data,  # dataset.yaml path
        imgsz=yolo_dim,  # inference size (height, width)
        conf_thres=confidence_thres,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=target_folder,  # save results to project/name
        name=video_name,  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,) # video frame-rate stride)

    return os.path.join(target_folder,video_name,"crops","dog_face")
       
#function to join the cropped frames back to video

def join_frames(h_out,w_out, frames_loc,target_folder,old_video_name,new_video_name, video_path, frame_rate=30):
    dim=(int(w_out),int(h_out)) #set up the output dimension of the videos
    
    try:
        #to get to the folder having the cropped frames of the video
        frame_list=os.listdir(frames_loc)
        if len(frame_list)<16:
            raise FileNotFoundError
        frame_list= sorted(frame_list,key=lambda x: int(re.sub('\D',"",x)))# must sort the list since the frames order could be shuffled up
        
        # initiate cv2 video wirter
        writer=cv2.VideoWriter(os.path.join(target_folder,new_video_name),cv2.VideoWriter_fourcc(*'X264'),frame_rate, dim)
        #loop all frames to resize and use writer to write them into a list
        for frame in frame_list:
            curr_frame=cv2.imread(os.path.join(frames_loc,frame))
            curr_frame=cv2.resize(curr_frame,dim,interpolation=cv2.INTER_CUBIC)
            #append the frames to a list
            writer.write(curr_frame)

        #release the writer object
        writer.release()
    
    except FileNotFoundError:
        #no cropping found, and just copy the original video
        to_copy=os.path.join(video_path,old_video_name)
        shutil.copy(to_copy,os.path.join(target_folder,new_video_name))

        
def shorten_width(raw_video_folder, target_folder, crop_ratio=0.8):
    raw_videos=os.listdir(raw_video_folder)

    for vid in tqdm(raw_videos):
        video=os.path.join(raw_video_folder,vid)
        cap=cv2.VideoCapture(video)
        # define the output video resolution
        h_out=int(cap.get(4))
        ori_width=int(cap.get(3))
        w_out=int(ori_width*crop_ratio)
        frame_rate=cap.get(5)

        out=cv2.VideoWriter(os.path.join(target_folder,vid),cv2.VideoWriter_fourcc(*'XVID'),frame_rate,(w_out,h_out))

        if not cap.isOpened():
            print(f'failed to open {video}')
        while cap.isOpened():
            ret,frame=cap.read()
            if ret:
                total_cut=ori_width-w_out
                left=total_cut//2
                right=total_cut-left
                frame=frame[:,left:ori_width-right+1]
                out.write(frame)
            else:
                break
        cap.release()
        out.release()

def show_frames (video_path,sec=None):
    video=cv2.VideoCapture(video_path)
    fps=int(video.get(cv2.CAP_PROP_FPS))
    print(fps)

    #frame_id=int(fps*sec)
    #video.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
    ret,frame=video.read()
    count=0

    os.makedirs("./captured", exist_ok=True)

    while ret:
        if count%2==0:
            cv2.imwrite('./captured/frame{}.jpg'.format(count),frame)
        ret,frame=video.read()
        #print("the next frame:",ret)
        count+=1


def FCM_process(video_path,video_name,data,weight):
    """function to launch FCM for UI API"""
    h_out, w_out = 384, 384  # this is for the cropped video dimesions
    confidence_thres = 0.6
    yolo_dim = (384, 384)  # this is for yolo inference

    # make a new folder to save the cropped videos
    target_folder = os.path.join(video_path,'cropped_vids')
    os.makedirs(target_folder,exist_ok=True)

    try:
        # to run the detect model per video, the output is the abs dir of the cropped images
        frames_loc = crop_videos(video_path,
                                 video_name,
                                 weight,
                                 data=data,
                                 confidence_thres=confidence_thres,
                                 target_folder=target_folder,
                                 yolo_dim=yolo_dim)

        # to join back all the cropped frames
        join_frames(h_out, w_out, frames_loc, video_path, video_name,"fcm_"+video_name, video_path, frame_rate=30)
    finally:
        shutil.rmtree(target_folder)


# if __name__== "__main__":
#     # #here we define all hyperparameters to run the Yolo
#     # h_out,w_out=384,384 #this is for the cropped video dimesions
#     # confidence_thres=0.6
#     # yolo_dim=(384,384) #this is for yolo inference
#     # #yaml file path
#     # data='./data/custom.yaml'
#     # #trained model parameters
#     # weight='./best.pt'
#     # #make a new folder to save the cropped videos
#     # target_folder='cropped_vids'
#     # joined_folder='joined_vids'
#     # os.makedirs('./'+target_folder,exist_ok=True)
#     # os.makedirs('./'+joined_folder,exist_ok=True)
#     #
#     # video_path=os.path.join(os.getcwd(),"videos")
#     # #first time run to shorten the width
#     # #shorten_width("./raw_videos",video_path)
#     #
#     # video_list= os.listdir(video_path)
#     #
#     # #start scanning through all videos and crop them
#     # for vid in tqdm(video_list):
#     #     #to run the detect model per video
#     #     frames_loc=crop_videos(video_path,
#     #     vid,weight,
#     #     data=data,
#     #     confidence_thres=confidence_thres,
#     #     target_folder=target_folder,
#     #     yolo_dim=yolo_dim)
#     #
#     #     #to join back all the cropped frames
#     #     join_frames(h_out,w_out, frames_loc,joined_folder,vid,vid, video_path,frame_rate=30)
#       # show_frames('./raw_videos/c7f.mp4')
if __name__== "__main__":
    #yaml file
    config='./final_model/fcm/custom.yaml'
    weight='./final_model/fcm/best.pt'

    path=os.path.abspath('./static')
    FCM_process(path, 'raw1.mp4',config,weight)