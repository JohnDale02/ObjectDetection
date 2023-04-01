import jetson.inference
import jetson.utils

net = jetson.inferece.detectNet("ssd-mobilenet-v2", threshold=0.5)
# Createing net with pretrained ssd-mobilenet-v2 
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    detections = net.Detect(img, width, height)
    display.RenderOnce(img, width, height)
    display.SetTitle("Object Detection Network {:.0f} FPS".format(net.GetNetworkFPS()))  