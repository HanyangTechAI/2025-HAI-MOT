from ultralytics import YOLO

# YAML Configuration file
# data = "./datasets/VOC.yaml"
data = "./datasets/VisDrone.yaml"

model = YOLO("./weights/yolo11n.pt")
print("\n-----------Validate Original Model-----------\n")
metrics = model.val(data=data, plots=True)

print("\n-----------Validate 50 Epoch Model-----------\n")
# The best model after 50 Epoch
trained_model = YOLO("yolo11n-trained-50.pt")
metrics = trained_model.val(data=data, plots=True)

# Train for 100 Epoch Model
print("\n-----------Start Train-----------\n")
train_results = trained_model.train(
    data=data,
    epochs=50, # 50 more epoch on top of 50 epoch model, so 100 epoch 
    imgsz=640,  
    device="cuda", 
)

trained_model.save("yolo11n-trained-100.pt")