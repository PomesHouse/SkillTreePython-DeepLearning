import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import IPython

# russian plate 사용 plate finder
def plate_detector(img, plate_cascade_name): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    plate_cascade_name = plate_cascade_name
    plate_model = cv2.CascadeClassifier()
    plate_model.load(cv2.samples.findFile(plate_cascade_name))
    pred = plate_model.detectMultiScale(hist)
    x, y, w, h = pred[0]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2_imshow(img)
        

##############################################
def set_model(weight_file, cfg_file):
    model = cv2.dnn.readNet(weight_file, cfg_file)
    predict_layer_names = [model.getLayerNames()[i[0]-1] for i in model.getUnconnectedOutLayers()]
    # ['yolo_82', 'yolo_94', 'yolo_106']
    return model, predict_layer_names
###############################################

def set_label(name_file):
    with open(name_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names
################################################

def get_predicts(img, model, predict_layer_names, min_confidence = 0.5):
    img_h, img_w, img_c = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    preds_each_layers = model.forward(predict_layer_names)

    boxes = []
    confidences=[]
    class_ids = []
    for preds in preds_each_layers:
        for pred in preds:
            box, confidence, class_id = pred[:4], pred[4], np.argmax(pred[5:])
            if confidence > min_confidence:
                x_center, y_center, w, h = box
                x_center, w = int(x_center*img_w), int(w*img_w)
                y_center, h = int(y_center*img_h), int(h*img_h)
                x, y = x_center-int(w/2), y_center-int(h/2)
        
                boxes.append([x, y, w, h])
                confidences.append(float(confidence)) # float 처리를 해야 NMSBoxes 함수 사용 가능
                class_ids.append(class_id)
    return boxes, confidences, class_ids
#################################################################

def crop_result(img, boxes, confidences, class_ids, class_names, plate_cascade_name, min_confidence=0.5):
    selected_box_idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for bi, (x, y, w, h) in enumerate(boxes):
        if bi in selected_box_idx:
            class_id = class_ids[bi]
            #color = class_colors[class_id]
            class_name = class_names[class_id]
        
            if class_name == 'car':
                if x < 0:
                    x = 0 
                if  y < 0:
                    y = 0
                #cv2.rectangle(img, (x, y), (x+w, y+h), color , 2)
                #cv2.putText(img, class_name, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2 )
                cropped = img[y:y+h, x:x+w]
                #cv2_imshow(cropped)
                plate_detector(cropped, plate_cascade_name)

def plate2detect(img, model, predict_layer_names, class_names, plate_cascade_name, min_confidence = 0.5):
    boxes, confidences, class_ids = get_predicts(img, model, predict_layer_names, min_confidence = 0.5)
    crop_result(img, boxes, confidences, class_ids, class_names, plate_cascade_name, min_confidence = min_confidence)
