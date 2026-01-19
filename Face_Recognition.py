import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import os 



Know_embed=[]
Know_name=[]

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
            
            encodings=face_recognition.face_encodings(
                    rgb,
                    known_face_locations=[(top,right,bottom,left)]
                )
            
            if not encodings :
                continue
            
            current_embedding = encodings[0]
            
            distance=face_recognition.face_distance(
                Know_embed,
                current_embedding
            )
            
            best_index=np.argmin(distance)
            best_distance=distance[best_index]
            
            Threshold=0.4
            
            if best_distance <Threshold:
                name=Know_name[best_index]
            else:
                name="Unknown"
                
            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
                )
        cv2.imshow("Enrollment",frame)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

                
            