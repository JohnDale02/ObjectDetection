import jetson_inference
import jetson_utils
import sys
import argparse

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=jetson_inference.poseNet.Usage() + jetson_utils.videoSource.Usage() + jetson_utils.videoOutput.Usage() + jetson_utils.Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson_inference.poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
camera = jetson_utils.gstCamera(1280, 720, "/dev/video0")
display = jetson_utils.glDisplay()

# process frames until EOS or the user exits
while display.IsOpen():
    img, width, height = camera.CaptureRGBA()

    if img is None: # timeout
        continue  

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)

    # render the image
    display.RenderOnce(img, width, height)

    # update the title bar
    display.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

