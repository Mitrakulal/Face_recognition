import cv2 
import mediapipe as mp
import face_recognition 
import os
import numpy as np

mp_face_detection=mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
enroll=False
stable_count=0
required_stable=15
move_thresh=10
MAX_em=10

prev_pos=None
embeddings=[]

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
    
        if enroll and results.detections:
                
                
            det=results.detections[0]   
            bbox=det.location_data.relative_bounding_box
            h,w,_=frame.shape
                
            x=int(bbox.xmin*w)
            y=int(bbox.ymin*h)
            box_w=int(bbox.width*w)
            box_h=int(bbox.height*h)
            
            if prev_pos:
                dx=abs(x-prev_pos[0])
                dy=abs(y-prev_pos[1])
                stable_count=stable_count+1 if dx< move_thresh and dy < move_thresh else 0
            
            else:
                stable_count=1
                
            prev_pos=(x,y)

            cv2.rectangle(frame,
                          (x,y),
                          (x+box_w,
                           y+box_h),
                          (0,255,0),
                          2)
            
            cv2.putText(frame,
                        f"stable :{stable_count}",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                          2)
            
            if stable_count>=required_stable and len(embeddings)<MAX_em:
                top=y
                right=x+box_w
                bottom=y+box_h
                left=x
                
                top = max(0, top)
                left = max(0, left)
                right = min(w, right)
                bottom = min(h, bottom)
                
                if right - left < 20 or bottom - top < 20:
                    continue
                
                encodings=face_recognition.face_encodings(
                    rgb,
                    known_face_locations=[(top,right,bottom,left)]
                )
                
                if encodings:
                    embeddings.append(encodings[0])
                    print(f"captured embedding{len(embeddings)}")
                    
                stable_count=0
                
                if len(embeddings) == MAX_em :
                    os.makedirs("embeddings", exist_ok=True)
                    np.save(f"embeddings/{person_name}.npy", embeddings)
                    print("[INFO] Enrollment completed")
                    enroll = False
                    embeddings = []
                    
        cv2.imshow("Enrollment",frame)
                
        key=cv2.waitKey(1) & 0xFF
                
        if key == ord('e'):
                    person_name=input("Enter name for enrollment :").strip().lower()
                    enroll=True
                    embeddings=[]
                    stable_count=0
                    prev_pos=None
                
        if key == ord('q'):
                    break
                
                    
                    
                    
            
                
                
                
            
            
               
                # cv2.putText(frame,
                #             str(detection.score[0]),
                #             (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,
                #             0.6,
                #             (0,255,0),
                #             2)
                # cv2.rectangle(frame,
                #               (x,y),
                #               (x+box_w,y+box_h),
                #               (0,255,0),
                #               2
                #               )
                
                # print("Enrollment or recoginition(E/r)?")
                # str=input().lower()
                
                # if str=="e":
                #     print("captured")
                    
                    
                    
        

            
        # cv2.imshow("MediaPipe Face Detection", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
cap.release()
cv2.destroyAllWindows()

    
