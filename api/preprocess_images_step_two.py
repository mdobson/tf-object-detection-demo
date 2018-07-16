from PIL import Image
import os

input_directory = 'preprocessed-images-step-two'
output_directory = 'test-data'

for fh in os.listdir(input_directory):
	key = 0
	if fh != '.DS_Store':
		img = Image.open(os.path.join(input_directory, fh))	
		width, height = img.size
		iwidth = int(width)
		iheight = int(height * 0.80)
		for i in range(0, height, iheight):
			for j in range(0, width, iwidth):
				if i == 0:
					box = (j, height - iheight, j + iwidth, height)
					cropped_im = img.crop(box)
					cropped_im.save(os.path.join(output_directory, "{}_crop_{}".format(key, fh)))
					key += 1
					
				


