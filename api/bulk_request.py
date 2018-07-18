import requests as r
from PIL import Image
import os
from io import BytesIO
import csv

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

def send_request_to_image_api(i):
	buf = BytesIO()
	i.save(buf, 'jpeg')
	buf.seek(0)
	send_file = [
		('images', ('test.jpg', buf, 'image/jpg'))
	]
	response = r.post('http://localhost:5000/predict/image', files=send_file) 
	return response

def send_request_to_image_api_for_json(i):
	buf = BytesIO()
	i.save(buf, 'jpeg')
	buf.seek(0)
	send_file = [
		('images', ('test.jpg', buf, 'image/jpg'))
	]
	response = r.post('http://localhost:5000/predict/image', files=send_file, headers={'accept': 'application/json'}) 
	return response


def save_response(path, response):
	with open(path, 'wb') as f:
		for chunk in response:
			f.write(chunk)

return_data = [['file', 'class', 'score']]
def main(input_directory, output_directory):
	key = 0
	for fh in os.listdir(input_directory):
		if fh != '.DS_Store':
			image = Image.open(os.path.join(input_directory, fh))
			image_api_response = send_request_to_image_api(image)
			#json_image_api_response = send_request_to_image_api_for_json(image)	
			save_response(os.path.join(output_directory, "analyzed_{}".format(fh)), image_api_response)
			# json_data = json_image_api_response.json()
			# for entry in json_data:
			# 	return_data.append([fh, entry['class'], entry['score']])
	
	# with open('fidelity_run_2.csv', 'w') as csvfile:
	# 	csvwriter = csv.writer(csvfile)
	# 	for row in return_data:
	# 		csvwriter.writerow(row)	


main('image-blocks', 'analyzed-data')
