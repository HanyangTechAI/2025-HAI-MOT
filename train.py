from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# YAML Configuration file
# data = "./datasets/VOC.yaml"
data = "./datasets/VisDrone.yaml"

train_results = model.train(
    data=data,
    epochs=50,  
    imgsz=640,  
    device="cuda", 
)

metrics = model.val()

results = model("./datasets/test_image.jpg")
results[0].show()  

model.save("yolo11n-trained.pt")