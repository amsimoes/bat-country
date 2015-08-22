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
ap.add_argument("-it","--iter-n", required=False, type=int, help = "Number of iterations of the layer (default = 10)")
ap.add_argument("-oc","--octave-n", required=False, type=int, help = "Number of octaves (default = 4)")
ap.add_argument("-s","--octave-scale", required=False, type=float, help = "Scale of the octaves (default = 1.4)")
ap.add_argument("-i", "--image", required = True, help = "path to image file")
ap.add_argument("-o", "--output", required = True, help = "path to output directory")
args = ap.parse_args()
 
# filter warnings, initialize bat country, and grab the layer names of
# the CNN
warnings.filterwarnings("ignore")
bc = BatCountry(args.base_model,args.proto,args.caffe_model)
layers = bc.layers()
 
# extract the filename and extension of the input image
filename = args.image[args.image.rfind("/") + 1:]
(filename, ext) = filename.split(".")
 
# loop over the layers -- VISUALIZING ALL LAYERS of the model
for (i, layer) in enumerate(layers):
	# perform visualizing using the current layer
	print("[INFO] processing layer `{}` {}/{}".format(layer, i + 1, len(layers)))
 
	try:
		# pass the image through the network
		image = bc.dream(np.float32(Image.open(args.image)),iter_n=int(args.iter_n), octave_n=int(args.octave_n),end=layer,verbose=False)
 
		# draw the layer name on the image
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		cv2.putText(image, layer, (5, image.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
 
		# construct the output path and write the image to file
		p = "{}/{}_{}.{}".format(args.output, filename, layer, ext) # instead of the pic index, writes the layer used
		cv2.imwrite(p, image)
 
	except KeyError, e:
		# the current layer can not be used
		print("[ERROR] cannot use layer `{}`".format(layer))
 
# perform housekeeping
bc.cleanup()