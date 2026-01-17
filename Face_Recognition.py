import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import os 
from collections import Counter


Know_embed=[]
Know_name=[]

Threshold=0.4

Frame_interval=5
hist_size=7
name_hist=[]
frame_count=0
last_name="Unknown"

EMBEDDINGS_DIR="embeddings"

for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".npy"):
        name =os.path.splitext(file)[0]
        data=np.load(os.path.join(EMBEDDINGS_DIR,file))
        
        if data.ndim==1:
            Know_embed.append(data)
            Know_name.append(name)
        else:
            for emb in data:
                Know_embed.append(emb)
                Know_name.append(name)
                
# def recognize(
#                 frame,
#                 bbox,
#                 Known_embed,
#                 Known_name
                
#               ):
   
def update_identity(current_name):
    global name_hist,last_name
    
    if current_name is None :
        return last_name
    
    name_hist.append(current_name)
    
    if len(name_hist)>hist_size:
        name_hist.pop(0)
        
    stable_name= Counter(name_hist).most_common(1)[0][0]
    
    last_name=stable_name
    
    return last_name
    
             
mp_face_detection=mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
    ) as face_detection:
    
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Camera not opened")
        exit()
    else:
        print(" Camera opened")

    while True:
        ret, frame = cap.read()
        frame_count +=1
        if not ret:
            continue
    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        
        results = face_detection.process(rgb)
    
        if  results.detections:
            det=results.detections[0]   
            bbox=det.location_data.relative_bounding_box
            h,w,_=frame.shape
                
            x=int(bbox.xmin*w)
            y=int(bbox.ymin*h)
            box_w=int(bbox.width*w)
            box_h=int(bbox.height*h)
            
            cv2.rectangle(frame,
                          (x,y),
                          (x+box_w,
                           y+box_h),
                          (0,255,0),
                          2)
            
            top=y
            right=x+box_w
            bottom=y+box_h
            left=x
            
            top = max(0, top)
            left = max(0, left)
            right = min(w, right)
            bottom = min(h, bottom)
            current_name=None
        

            if frame_count % Frame_interval ==0:
                
                encodings=face_recognition.face_encodings(
                        rgb,
                        known_face_locations=[(top,right,bottom,left)]
                    )
                
                if encodings :
                    
                
                    current_embedding = encodings[0]
                    
                    distance=face_recognition.face_distance(
                        Know_embed,
                        current_embedding
                    )
                    
                    best_index=np.argmin(distance)
                    best_distance=distance[best_index]
                    
                    
                    
                    if best_distance <Threshold:
                        current_name=Know_name[best_index]
                    else:
                        current_name="Unknown"
                else:
                    current_name="Unknown"
            stable_name=update_identity(current_name)
            cv2.putText(
                frame,
                stable_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
                )
        else:
            stable_name = update_identity(None)
                
        cv2.imshow("Enrollment",frame)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

                
            