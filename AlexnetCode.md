```
%%capture
!pip install wandb
!apt-get install poppler-utils
!pip install pdf2image
!pip install flashtorch
import requests
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import numpy as np
import torch
import requests
from torchvision import *
from torchvision.models import *
import wandb as wb
```
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=device)

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=device)

def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(5, 5)
    plt.show()

def get_google_slide(url):
    url_head = "https://docs.google.com/presentation/d/"
    url_body = url.split('/')[5]
    page_id = url.split('.')[-1]
    return url_head + url_body + "/export/pdf?id=" + url_body + "&pageid=" + page_id

def get_slides(url):
    url = get_google_slide(url)
    r = requests.get(url, allow_redirects=True)
    open('file.pdf', 'wb').write(r.content)
    images = convert_from_path('file.pdf', 500)
    return images

def load(image, size=224):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    tensor.requires_grad = True
    return tensor
```
```
labels = {int(key):value for (key, value) in requests.get('https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json').json().items()}

model = alexnet(weights='DEFAULT').to(device)
model.eval();
```
```
url = "https://docs.google.com/presentation/d/197FrV3VQS7epab36L3JHWUw1E5opGGDU-xeTYVpOcOY/edit#slide=id.g1e5fe554cc7_0_0"
```
```
images = []

for image in get_slides(url):

    plot(image)

    images.append(load(image))

images = torch.vstack(images)
```
```
y = model(images)
```
```
guesses = torch.argmax(y, 1).cpu().numpy()
```
```
for i in list(guesses):
    print(labels[i])
```
```
Y = np.zeros((100),)
Y[50:] = 1
```
```
X = y.detach().cpu().numpy()
```
```
plt.plot(X[0],'.')
```
![Unknown-17](https://github.com/Carlbronge/IonDetect-Innovations/assets/143009718/5f31a264-37f9-4a9b-baf2-54d5735ddf0a)
```
plt.hist(X[0])
```
![Unknown-18](https://github.com/Carlbronge/IonDetect-Innovations/assets/143009718/7ccfe7f8-6935-49cc-885f-db41e2277e2f)
```
X = GPU_data(X)
Y = GPU_data(Y)
```
```
def softmax(x):
    s1 = torch.exp(x - torch.max(x,1)[0][:,None])
    s = s1 / s1.sum(1)[:,None]
    return s
```
```
def cross_entropy(outputs, labels):
    return -torch.sum(softmax(outputs).log()[range(outputs.size()[0]), labels.long()])/outputs.size()[0]
```
```
def randn_trunc(s): #Truncated Normal Random Numbers
    mu = 0
    sigma = 0.1
    R = stats.truncnorm((-2*sigma - mu) / sigma, (2*sigma - mu) / sigma, loc=mu, scale=sigma)
    return R.rvs(s)
```
```
def Truncated_Normal(size):

    u1 = torch.rand(size)*(1-np.exp(-2)) + np.exp(-2)
    u2 = torch.rand(size)
    z  = torch.sqrt(-2*torch.log(u1)) * torch.cos(2*np.pi*u2)

    return z
```
```
def acc(out,y):
    with torch.no_grad():
        return (torch.sum(torch.max(out,1)[1] == y).item())/y.shape[0]
```
```
a = np.array([[1, 2], [3, 4]])
```
```
b = np.array([[5, 6], [1, 0]])
```
```
def inner_prod(a,b):
    c = 0
    for i in range(a.shape[0]):
        c += a[i]*b[i]
    return c
```
```
def mat_mult(a,b):

    if a.shape[1] == b.shape[0]:

        c = np.zeros((a.shape[0],b.shape[1]))

        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
               c[i,j] = inner_prod(a[i,:],b[:,j])
        return c

    else:
        print("error")
```
```
%%timeit
a@b
```
```
%%timeit
mat_mult(a,b)
```
```
x = np.random.random((1000,1))
```
```
w = np.random.random((2,1000))
```
```
def get_batch(mode):
    b = c.b
    if mode == "train":
        r = np.random.randint(X.shape[0]-b)
        x = X[r:r+b,:]
        y = Y[r:r+b]
    elif mode == "test":
        r = np.random.randint(X_test.shape[0]-b)
        x = X_test[r:r+b,:]
        y = Y_test[r:r+b]
    return x,y
```
```
def model(x,w):

    return x@w[0]
```
```
def make_plots():

    acc_train = acc(model(x,w),y)

    # xt,yt = get_batch('test')

    # acc_test = acc(model(xt,w),yt)

    wb.log({"acc_train": acc_train})
```
```
wb.init(project="Linear_Model_Photo_1");
c = wb.config

c.h = 0.001
c.b = 32
c.epochs = 100000

w = [GPU(Truncated_Normal((1000,2)))]

optimizer = torch.optim.Adam(w, lr=c.h)

for i in range(c.epochs):

    x,y = get_batch('train')

    loss = cross_entropy(softmax(model(x,w)),y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wb.log({"loss": loss})

    make_plots()
```
