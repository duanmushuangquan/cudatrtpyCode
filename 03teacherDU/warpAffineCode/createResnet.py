# renet 训练的时候使用的图片大小为224 * 224
# 使用的均值标准差是：
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
import torchvision.models as models
import torchvision.transforms as T
import cv2
import torch.nn as nn
import torch 

# 1.导入模型
model = models.resnet18(pretrained=True)
model.eval()
model.cuda()

# 2.1 均值、标准差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 这里默认认为输入图片是224*224的。就没有resize，主要是考虑到后面需要和c++对应
trans = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std), 
    T.Lambda(lambda x : x.unsqueeze(dim=0).cuda())
])

image = cv2.imread("kj.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

torch_image = trans(image)

with torch.no_grad():
    predict = model(torch_image)
prob = torch.softmax(predict, dim=1)
label = torch.argmax(prob)
confidence = prob[0, label]

with open("labels.imagenet.txt", "r") as f:
    labels = f.readlines()
print(f"类别={label}, 中文名={labels[label].strip()}, 置信度{confidence}")


# import torch.onnx
# torch.onnx.export(model, 
#                   (torch_image,), 
#                   "resnet18.onnx", 
#                   input_names=["image"], 
#                   output_names=["Predict"],
#                   opset_version=11,
#                   verbose=True)