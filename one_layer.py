# import the necessary packages
from __future__ import print_function
from batcountry import BatCountry
from PIL import Image
import numpy as np
import argparse
import warnings
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--base-model", required=True, help = "base model path")
ap.add_argument("-p", "--proto", required=False, help = "prototxt model path (default = 'deploy.prototxt')")		# added for the specific CNN prototxt
ap.add_argument("-m", "--caffe-model", required=False, help= "caffe model path (default = 'imagenet.caffemodel')")	# added for custom caffe-model
ap.add_argument("-it","--iter-n", required=False, default=10, type=int, help = "Number of iterations of the layer (default = 10)")
ap.add_argument("-oc","--octave-n", required=False, default=4, type=int,help = "Number of octaves (default = 4)")
ap.add_argument("-s","--octave-scale", required=False, type=float, default=1.4, help = "Scale of the octaves (default = 1.4)")
ap.add_argument("-l","--layer", required=False, type=str, default="conv3", help = "layer of CNN to use")
ap.add_argument("-i", "--image", required = True, help = "path to image file")
ap.add_argument("-o", "--output", required = True, help = "path to output directory")
args = ap.parse_args()

# filter warnings, initialize bat country, and grab the layer names of
# the CNN
warnings.filterwarnings("ignore")
bc = BatCountry(args.base_model,args.proto,args.caffe_model)

print("[INFO] processing layer `{}`".format(args.layer))
image = bc.dream(np.float32(Image.open(args.image)), iter_n=args.iter_n, octave_n=args.octave_n, end=args.layer)
 
# extract the filename and extension of the input image
filename = args.image[args.image.rfind("/") + 1:]
(filename, ext) = filename.split(".")

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.putText(image, args.layer, (5, image.shape[0] - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
 
# construct the output path and write the image to file
p = "{}/{}_{}_{}.{}".format(args.output, filename, args.layer, args.iter_n, ext) # instead of the pic index, writes the layer used
cv2.imwrite(p, image)

bc.cleanup();