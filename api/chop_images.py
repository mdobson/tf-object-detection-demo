from PIL import Image

im = Image.open('resized2.jpg')
width, height = im.size
iwidth = int(width)
iheight = int(height/2)
for i in range(0, height, iheight):
	for j in range(0, width, iwidth):
		box = (j, i, j+iwidth, i+iheight)
		cropped_im = im.crop(box)
		cropped_im.save("crop3_{}_{}.jpg".format(i,j))
		print('[{},{}]'.format(i, j)) 
