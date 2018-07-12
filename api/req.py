from PIL import Image
import requests as r
from io import BytesIO

i = Image.open('experiments/test1.jpg')
buf = BytesIO()
i.save(buf, 'jpeg')
buf.seek(0)
send_file = [
	('images', ('test.jpg', buf, 'image/jpg'))
]
response = r.post('http://localhost:5000/predict/image', files=send_file) 
print(response.status_code)

with open('test1.jpg', 'wb') as f:
	for chunk in response:
		f.write(chunk)
