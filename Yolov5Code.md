```
!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt comet_ml  # install

import torch
import utils
display = utils.notebook_init()  # checks
```
```
!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images
display.Image(filename='runs/detect/exp/zidane.jpg', width=600)
```
![Unknown](https://github.com/Carlbronge/IonDetect-Innovations/assets/143009718/7d11bccd-f85d-416f-a22d-94c3e9fd8d3c)
```
import requests
from PIL import Image
from io import BytesIO

url = "https://e00-marca.uecdn.es/assets/multimedia/imagenes/2023/09/19/16951019159405.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
image_path = "downloaded_image.jpg"
image.save(image_path)
```
```
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
```
```
results = model(image_path)
```
```
results.show()
```
![Unknown-19](https://github.com/Carlbronge/IonDetect-Innovations/assets/143009718/d0543b68-c2da-4478-816c-b4d027156873)
```
!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source https://www.youtube.com/watch?v=3kSnrwJRqW8
```


