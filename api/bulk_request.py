import requests as r
from PIL import Image
import os
from io import BytesIO

def resize_image(img_name):
	i = Image.open(img_name)
	h,w = i.size
	ri = i.resize((int(h/2), int(w/2)), Image.ANTIALIAS)
	return ri


def crop_image(img):
	width, height = img.size
	iwidth = int(width)
	iheight = int(height/2)
	for i in range(0, height, iheight):
		for j in range(0, width, iwidth):
			box = (j, i, j+iwidth, i+iheight)
			cropped_im = img.crop(box)
			yield cropped_im

def send_request_to_json_api(i):
	buf = BytesIO()
	i.save(buf, 'jpeg')
	buf.seek(0)
	send_file = [
		('images', ('test.jpg', buf, 'image/jpg'))
	]
	response = r.post('http://localhost:5000/predict/image', headers={'accept': 'application/json'}, files=send_file) 
	return response


def send_request_to_image_api(i):
	buf = BytesIO()
	i.save(buf, 'jpeg')
	buf.seek(0)
	send_file = [
		('images', ('test.jpg', buf, 'image/jpg'))
	]
	response = r.post('http://localhost:5000/predict/image', files=send_file) 
	return response

def save_response(path, response):
	with open(path, 'wb') as f:
		for chunk in response:
			f.write(chunk)


def main(input_directory, output_directory):
	key = 0
	for fh in os.listdir(input_directory):
		if fh != '.DS_Store':
			image = Image.open(os.path.join(input_directory, fh))
			image_api_response = send_request_to_image_api(image)	
			save_response(os.path.join(output_directory, "analyzed_{}".format(fh)), image_api_response)


main('test-data', 'analyzed-data')
