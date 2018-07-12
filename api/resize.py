from PIL import Image

i = Image.open('test2.jpg')

h,w = i.size

ri = i.resize((int(h/2), int(w/2)), Image.ANTIALIAS)

ri.save('resized2.jpg', FORMAT='JPEG')
