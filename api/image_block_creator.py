from PIL import Image
import os

input_directory = 'preprocessed-images-step-two'
output_directory = 'image-blocks'

for fh in os.listdir(input_directory):
  key = 0
  if fh != '.DS_Store':
    img = Image.open(os.path.join(input_directory, fh))	
    width, height = img.size
    iwidth = int(width/2)
    iheight = int(height/2)
    for i in range(0, height, iheight):
      for j in range(0, width, iwidth):
        box = (j, i, j+iwidth, i+iheight)
        cropped_img = img.crop(box)
        cropped_img.save(os.path.join(output_directory,"cropof_{}_{}_{}.jpg".format(fh,i,j))) 
    
