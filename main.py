from ultralytics import YOLO

model = YOLO("yolo11n.pt")  
results = model.train(data="./Flower/flower.yaml", epochs=100)

#afer trainning, trained model will be saved at runs/detect/train../weights

# model = YOLO("runs/dectect/train5/weights/best.pt") 
# te = model("torenia_01.jpg",save=True)
