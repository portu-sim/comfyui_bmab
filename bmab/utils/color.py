import cv2
import numpy as np
from skimage import exposure
from blendmodes.blend import blendLayers, BlendType
from PIL import Image


def apply_color_correction(correction, original_image):
	image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
		cv2.cvtColor(
			np.asarray(original_image),
			cv2.COLOR_RGB2LAB
		),
		cv2.cvtColor(np.asarray(correction.copy()), cv2.COLOR_RGB2LAB),
		channel_axis=2
	), cv2.COLOR_LAB2RGB).astype("uint8"))

	image = blendLayers(image, original_image, BlendType.LUMINOSITY)

	return image.convert('RGB')
