from ultralytics import YOLO

model = YOLO("yolo11n.pt")  

results = model.train(data="./Flower/flower.yaml", epochs=100)

te = model("torenia_01.jpg",save=True)
