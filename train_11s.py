from ultralytics import YOLO

# YAML Configuration file
data = "./datasets/VisDrone.yaml"

model = YOLO("./weights/yolo11s.pt")
print("\n-----------Validate Original Model-----------\n")
metrics = model.val(data=data, plots=True)

# Train for 100 Epoch Model
print("\n-----------Start Train-----------\n")
train_results = model.train(
    data=data,
    epochs=100, 
    imgsz=640,  
    device="cuda", 
)

model.save("yolo11s-trained-100.pt")