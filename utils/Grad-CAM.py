import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import random
from torch.backends import cudnn
from model import resnet18, resnet34, resnet50, resnet101


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def x_norm(x):
    x_norm = np.linalg.norm(x, ord=None, axis=1, keepdims=False).reshape(-1, 1)
    x = x / x_norm
    return x


setup_seed(42)
path = '/home/private/hrrp/NanHu/resnetV2/data/j机/test/Y20.csv'
xy = pd.read_csv(path, header=None)
xy = xy
x = xy.iloc[:, :-1].values
X_test = x_norm(x)
label = xy.iloc[:, -1]
enc = LabelEncoder()  # 获取一个LabelEncoder
labels = ['Prop1', 'Prop3', 'Prop4', 'Y20']
enc = enc.fit(labels)
y = enc.transform(label)


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations = output

    def generate(self, x, target_layer=None):
        if target_layer is None:
            target_layer = self.model.layers[-1]

        handle_forward = target_layer.register_forward_hook(self.forward_hook)
        handle_backward = target_layer.register_backward_hook(self.backward_hook)

        output = self.model(x.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        outputs = torch.softmax(output, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        print(outputs[0]+1)
        one_hot = F.one_hot(torch.tensor([pred]), num_classes=output.size()[1]).float().cuda()
        output.backward(gradient=one_hot)

        weights = F.adaptive_avg_pool1d(self.gradients, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        if len(cam.shape) < 3:
            cam = cam.unsqueeze(0)
        cam = F.interpolate(cam, size=x.shape[-1], mode='linear')

        handle_forward.remove()
        handle_backward.remove()

        return cam.squeeze()


# 选择一个测试样本
# idx = np.random.randint(len(X_test))
idx = 30
x = X_test[idx]
x = torch.from_numpy(x)
x = torch.unsqueeze(x.float(), dim=0)
y = torch.from_numpy(y).long().cuda()[idx]
print(y)

model = resnet18(5).cuda()
model_weight_path = '/home/private/hrrp/NanHu/resnetV2/weight/j机/best.pth'
model.load_state_dict(torch.load(model_weight_path))


# 生成Grad-CAM热力图
gradcam = GradCAM(model)
cam = gradcam.generate(x.cuda(), model.layer4).cpu().detach().numpy()
x = x.squeeze().cpu().detach().numpy()
# cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
# x = (x - x.min()) / (x.max() - x.min() + 1e-7)

plt.figure()
plt.plot(x)
plt.plot(cam)
plt.show()