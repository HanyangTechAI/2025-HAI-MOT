from ultralytics import YOLO

model = YOLO("yolo11n.pt")

train_results = model.train(
    data="./datasets/VOC.yaml",  # dataset configuration file
    epochs=10,  
    imgsz=640,  
    device="cuda", 
)

metrics = model.val()

results = model("./datasets/test_image.jpg")  # Predict on an image
results[0].show()  

# path = model.export(format="onnx")