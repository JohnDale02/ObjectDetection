import jetson_inference
import jetson_utils

net = jetson_inference.detectNet("Pose-DenseNet121-body", threshold=0.5)
# Createing net with pretrained densenet121-body human pose 
camera = jetson_utils.gstCamera(1280, 720, "/dev/video0")
display = jetson_utils.glDisplay()

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    detections = net.Detect(img, width, height)
    display.RenderOnce(img, width, height)
    display.SetTitle("Object Detection Network {:.0f} FPS".format(net.GetNetworkFPS()))  
