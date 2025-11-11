from ultralytics import YOLO

model = YOLO("yolo11n.pt")  
results = model.train(data="./Flower/flower.yaml", epochs=100)

#afer trainning, trained model will be saved at runs/detect/train../weights

model = YOLO("runs/detect/train6/weights/last.pt") 
te = model("flowerhh.jpg",save=True,conf=0.1)
