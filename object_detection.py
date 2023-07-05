import torch
import cv2
class ObjectDetection:
    def __init__(self,args): 
        self.args = args
        self.model = self.load_model()
        self.classes = self.model.names
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s',pretrained = True)
        return model
    
    def detect(self,frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        lables,cord =results.xyxyn[0][:,-1].to('cpu').numpy(),results.xyxyn[0][:,:-1].to('cpu').numpy()
        return lables ,cord

    def class_to_label(self,x):
        return self.classes[int(x)]
    
    def plot_boxes(self,results,frame,thickness):
        labels,cord = results
        n = len(labels)
        x_shape,y_shape = frame.shape[1],frame.shape[0]
        
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),thickness)
                font_size,_=cv2.getTextSize(self.class_to_label(labels[i]),cv2.FONT_HERSHEY_DUPLEX,0.6,thickness)
                widht,height=font_size
                cv2.rectangle(frame,(x1,y2),(x1+widht,y2-height+28),(255,0,0),-1)
                cv2.putText(frame,self.class_to_label(labels[i]),(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,255),thickness)
        return frame