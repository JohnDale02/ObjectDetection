import jetson_inference
import jetson_utils

net = jetson_inference.poseNet("densenet121-body", threshold=0.5)
# Createing net with pretrained densenet121-body human pose 
camera = jetson_utils.gstCamera(1280, 720, "/dev/video0")
display = jetson_utils.glDisplay()

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    #image = camera.Capture()
    poses = net.Process(img, overlay='box')

    display.RenderOnce(img, width, height)
    display.SetTitle("Pose Detection Network {:.0f} FPS".format(net.GetNetworkFPS()))  
