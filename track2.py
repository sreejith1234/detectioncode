import sys
import requests
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.utils.torch_utils import time_sync as time_synchronized #time_sync as
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import speech_recognition as sr
import pyaudio
import wave
import telegram_send
import threading
global isThreadBusy
isThreadBusy = False



audio_yes = 'audio/yes.wav'
audio_okay = 'audio/okay.wav'
audio_invalid = 'audio/beep-04.wav'

def play_audio(audioFile):
    chunk = 1024
    wf = wave.open(audioFile, 'rb')
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pa.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)

    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    pa.terminate()

### Added
class point:
     
    def __init__(self):
         
        self.x = 0
        self.y = 0
import numpy as np
from pathlib import Path

import datetime

from configparser import ConfigParser
configur = ConfigParser()
configur.read('conf.ini')

prev_ids_up = []

"""##Quad
p1X = configur.getint('quad_coordinates','p1X')
p1Y = configur.getint('quad_coordinates','p1Y')

p2X = configur.getint('quad_coordinates','p2X')
p2Y = configur.getint('quad_coordinates','p2Y')

p3X = configur.getint('quad_coordinates','p3X')
p3Y = configur.getint('quad_coordinates','p3Y')

p4X = configur.getint('quad_coordinates','p4X')
p4Y = configur.getint('quad_coordinates','p4Y')"""

#Line
p1xL = configur.getint('line_coordinates','p1X')
p1yL = configur.getint('line_coordinates','p1Y')

p2xL = configur.getint('line_coordinates','p2X')
p2yL = configur.getint('line_coordinates','p2Y')
line_start = (p1xL, p1yL)
line_end = (p2xL, p2yL)

line_hight = configur.getfloat('line_coordinates','hight')

#extra added

p1xL1 = configur.getint('line_coordinates1','p1X1')
p1yL1 = configur.getint('line_coordinates1','p1Y1')

p2xL1 = configur.getint('line_coordinates1','p2X1')
p2yL1 = configur.getint('line_coordinates1','p2Y1')
line_start1 = (p1xL1, p1yL1)
line_end1 = (p2xL1, p2yL1)

line_hight1 = configur.getfloat('line_coordinates1','hight1')

"""object_inside = str(configur.get('object_inside','object'))
#############Added

pts = np.array([[p1X, p1Y], [p2X, p2Y], 
                [p3X, p3Y], [p4X, p4Y]],
               np.int32)
ptsC= [(p1X, p1Y), (p2X, p2Y),(p3X, p3Y), (p4X, p4Y)]
  
pts = pts.reshape((-1, 1, 2))
  
isClosed = True
  
# Blue color in BGR
#color = (0, 255, 0)
  
# Line thickness of 2 px
#thickness = 2

###################Added is inside
# A Python3 program to check if a given point
# lies inside a given polygon
# Refer https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# for explanation of functions onSegment(),
# orientation() and doIntersect()

# Define Infinite (Using INT_MAX
# caused overflow problems)
INT_MAX = 10000

# Given three colinear points p, q, r,
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
    
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
        
    return False

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are colinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
    
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
        ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock

def doIntersect(p1, q1, p2, q2):
    
    # Find the four orientations needed for
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True
    
    # Special Cases
    # p1, q1 and p2 are colinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are colinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are colinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are colinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True

    return False

# Returns true if the point p lies
# inside the polygon[] with n vertices
def is_inside_polygon(points:list, p:tuple) -> bool:
    
    n = len(points)
    
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
        
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0
    
    while True:
        next = (i + 1) % n
        
        # Check if the line segment from 'p' to
        # 'extreme' intersects with the line
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):
                            
            # If the point 'p' is colinear with line
            # segment 'i-next', then check if it lies
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p,
                        points[next]) == 0:
                return onSegment(points[i], p,
                                points[next])
                                
            count += 1
            
        i = next
        
        if (i == 0):
            break
        
    # Return true if count is odd, false otherwise
    return (count % 2 == 1)"""
#######


#####Line


# Constant integers for directions
RIGHT = 1
LEFT = -1
TOP = 1
BOTTOM = -1
ZERO = 0


def directionOfPoint(A, B, P):

    
    global RIGHT, LEFT, ZERO
    
    # Subtracting co-ordinates of
    # point A from B and P, to
    # make A as origin
    B.x -= A.x
    B.y -= A.y
    P.x -= A.x
    P.y -= A.y
    

    # Determining cross Product
    cross_product = B.x * P.y - B.y * P.x
    

    # Return RIGHT if cross product is positive
    if (cross_product < 0):
        return RIGHT
        
    # Return LEFT if cross product is negative
    if (cross_product > 0):
        return LEFT
    return ZERO


def directionOfPointTB(A, B, P):

    
    global TOP, BOTTOM, ZERO
    
    # Subtracting co-ordinates of
    # point A from B and P, to
    # make A as origin
    B.x -= A.x
    B.y -= A.y
    P.x -= A.x
    P.y -= A.y
    

    # Determining cross Product
    cross_product = B.x * P.x - B.y * P.y
    

    # Return RIGHT if cross product is positive
    if (cross_product < 0):
        return TOP
        
    # Return LEFT if cross product is negative
    if (cross_product > 0):
        return BOTTOM
    return ZERO


#Added End
def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def send_telegram_msg():
    global isThreadBusy
    isThreadBusy = True
    telegram_send.send(messages=["Alert!!  :Vehicle entered into restrictes area"])
    play_audio(audio_invalid)
    isThreadBusy = False
      
def detect(opt):
    global prev_ids_up
    global isThreadBusy
    global boundaryLines


    current_ids_up = []

    A = point()
    B = point()
    P = point()

    A1 = point()
    B1 = point()
    P1 = point()
   
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, device=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    """link = "https://api.thingspeak.com/update?api_key=M1SFSUUQCZG29YIL&field1=0"
    off = requests.get(link)
    print(off.text)
    print("light off")"""


    for frame_idx, (path, img, im0s, vid_cap, extra_ignore) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            
            im0_copy = im0.copy()
            #Added
            cv2.line(im0, line_start, line_end, (0,255,0), 2)
            cv2.line(im0, line_start1, line_end1, (0,0,255), 2)
            #cv2.polylines(im0, [pts], isClosed, (0,255,0), 2)

            ######
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs, ids = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
                #current_ids_up = []

                
                # draw boxes for visualization
                
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        color = compute_color_for_id(id)

                        #Added                  
                        c1, c2 = (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3]))

                        xC = int(bboxes[0]) + int((int(bboxes[2]) - int(bboxes[0]))/2)
                        yC = int(bboxes[1]) + int((int(bboxes[3]) - int(bboxes[1]))/2)
                        
                        #pC = (xC,yC)

                        A.x = p1xL
                        B.x = p2xL
                        B.y = p2yL
                        P.x = xC
                        P.y = yC

                        A1.x = p1xL1
                        B1.x = p2xL1
                        B1.y = p2yL1
                        P1.x = xC
                        P1.y = yC
                        
                        
                        direction = directionOfPoint(A,B,P)
                        #print("direction",direction,id)

                        direction_line_red = directionOfPoint(A1,B1,P1)
                        #print("direction red",direction_line_red,id)
                        direction_line_green = directionOfPointTB(A,B,P)

                        print("direction green",direction_line_red,direction_line_green,id)
                        
                        #if id == 7:
                            #pass
                        if direction_line_red == LEFT and direction_line_green == BOTTOM:  #RED
                            print("current_ids_up",current_ids_up)
                            if id not in current_ids_up:
                                current_ids_up.append(id)
                                print("entered")
                                date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                Path("./runs/line_crossed").mkdir(parents=True, exist_ok=True)
                                img_sv = "./runs/line_crossed/"+str(label)+"-" + date_string + ".png"
                                cropped = im0_copy[c1[1]:c2[1],c1[0]:c2[0]]
                                cv2.imwrite(img_sv,cropped)
                                lab_obj_line = str(label)+ " crossed line red"
                                cv2.putText(im0, lab_obj_line, (0, 60), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
                                play_audio(audio_invalid)
                                current_ids_up.append(id)
                        elif direction_line_red == RIGHT and direction_line_green == TOP:
                            print("current_ids_up",current_ids_up)
                            if id in current_ids_up:
                                current_ids_up.remove(id)


                        

                        """direction1 = directionOfPoint(A1,B1,P1)

                        if(direction1 == 0 or direction1 == -1): #RED
                            current_ids_up.append(id)
                        else: 
                            if(id in prev_ids_up):
                                date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                Path("./runs/line_crossed").mkdir(parents=True, exist_ok=True)
                                img_sv = "./runs/line_crossed/"+str(label)+"-" + date_string + ".png"
                                cropped = im0_copy[c1[1]:c2[1],c1[0]:c2[0]]
                                cv2.imwrite(img_sv,cropped)
                                lab_obj_line = str(label)+ " crossed line red"
                                cv2.putText(im0, lab_obj_line, (0, 60), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
                                play_audio(audio_invalid)"""
                        

                        #Quad
                        """if(is_inside_polygon(ptsC,pC) and label.find(object_inside) != -1):
                            date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                            Path("./runs/inside_quad").mkdir(parents=True, exist_ok=True)
                            img_sv = "./runs/inside_quad/"+str(label)+"-" + date_string + ".png"
                            cropped = im0_copy[c1[1]:c2[1],c1[0]:c2[0]]
                            cv2.imwrite(img_sv,cropped)
                            lab_obj =  "Vehicle entered into RESTRICTED AREA"
                            cv2.putText(im0, lab_obj, (0, 30), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
                            cv2.polylines(im0, [pts], isClosed, (0,0,255), 2)
                            #if(not isThreadBusy):
                             #   thread1 = threading.Thread(target=send_telegram_msg, args=())"""
                              #  if thread1.isAlive():
                               #     thread1.join()
                                #if not thread1.isAlive():
                                 #   thread1.start()
                        ############


                        plot_one_box(bboxes, im0, color=color, label=label, line_thickness=2)

                        if save_txt:
                            # to MOT format
                            bbox_top = output[0]
                            bbox_left = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                               f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_top,
                                                           bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                    prev_ids_up = current_ids_up.copy()

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # if show_vid:
            #     cv2.imshow(p, im0)
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         raise StopIteration
            if show_vid:
                #display=cv2.resize(im0,(100%,50%))
                #resized_down = cv2.resize(im0,(300,200), interpolation= cv2.INTER_LINEAR)
                resized_up = cv2.resize(im0, (640,480))
                #cv2.imshow(p, resized_down )
                cv2.imshow(p, resized_up )
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    
    parser.add_argument('--source', type=str, default='video/left--right.mp4', help='source')# file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
