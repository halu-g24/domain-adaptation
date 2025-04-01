import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
from PIL import Image
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import math

#Office31
source_path = './Office31/AtoD/source_images/'
target_path = './Office31/AtoD/target_images/'
knw_class_num = 10

#OfficeHome
#source_path = './OfficeHome/CltoPr/source_images/'
#target_path = './OfficeHome/CltoPr/target_images/'
#knw_class_num = 25

class TemplateDataset(Dataset):
    
    def __init__(self, path):
        self.path = path

        file_list = []
        for data in os.listdir(path):
            file_list.append(data)

        self.file_list = file_list
        
            
    def __len__(self):
        return len(self.file_list)
        
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.file_list[idx])
        image = Image.open(img_name)
        toten = transforms.ToTensor()
        image = toten(image)
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        image = norm(image)
        label = int(img_name.split('_')[-5])
        if label >= knw_class_num:
            label = knw_class_num

        return image, label
    

#特徴抽出器
class M(nn.Module):
    def __init__(self):
        super(M,self).__init__()
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*[x for x in list(resnet50.children())[:-1]]) # Upto the avgpool layer

    def forward(self,x):
        feats = self.features(x)
        return feats.view((x.shape[0], 2048))

#分類器
class C(nn.Module):
    def __init__(self):
        super(C,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256,knw_class_num+1)
        )
    def forward(self,x,reverse=False):
        if reverse:
            x = GradientReversalLayer.apply(x, 1)
        logits = self.classifier(x)
        return logits
    
#勾配反転層
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(context, x, constant):
        context.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(context, grad):
        return grad.neg() * context.constant, None
    
    
#gpu check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_G = M().to(device)
model_C = C().to(device)

criterion_CE = nn.CrossEntropyLoss()
criterion_BCE = nn.BCELoss()

#エポック数
epoch = 300

#バッチサイズ
batch_size = 32

#学習率
lr = 0.0002

#OSBPのハイパーパラメータ
#未知の正解ラベル
t = 0.5

#データローダー
dataset_source = TemplateDataset(source_path)
loader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
dataset_target = TemplateDataset(target_path)
loader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)

#オプティマイザー
optimizer_G = optim.SGD(model_G.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_C = optim.SGD(model_C.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

print(f'ソースデータ数: {len(dataset_source)}')
print(f'ターゲットデータ数: {len(dataset_target)}')
batch_roop = max(len(loader_source),len(loader_target))
print(f'バッチループ数: {batch_roop}')

for i in range(epoch):
    #学習
    model_G.train()
    model_C.train()

    #各loss
    train_loss_s = 0
    train_loss_adv = 0

    source_iter = iter(loader_source)
    target_iter = iter(loader_target)
    # 進行状況バーを作成
    progress_bar = tqdm(total=batch_roop)
    for _ in range(batch_roop):
        try:
            source_images, source_labels = next(source_iter)
            if source_images.shape[0] < batch_size:
                source_iter = iter(loader_source)
                source_images, source_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(loader_source)
            source_images, source_labels = next(source_iter)
        try:
            target_images, target_labels = next(target_iter)
            if target_images.shape[0] < batch_size:
                target_iter = iter(loader_target)
                target_images, target_labels = next(target_iter)
        except StopIteration:
            target_iter = iter(loader_target)
            target_images, target_labels = next(target_iter)

        optimizer_G.zero_grad()
        optimizer_C.zero_grad()

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)
        target_labels = target_labels.to(device)

        #loss_s
        s_outputs_G = model_G(source_images)
        s_outputs_C = model_C(s_outputs_G)
        loss_s = criterion_CE(s_outputs_C, source_labels)
        train_loss_s += loss_s.item()
        loss_s.backward(retain_graph=True)

        #loss_adv
        t_outputs_G = model_G(target_images)
        t_outputs_C = model_C(t_outputs_G, reverse=True)
        sm_t_outputs_C = F.softmax(t_outputs_C, dim=1)
        unk_class_p = sm_t_outputs_C[:,-1]
        unk_labels = torch.full_like(unk_class_p, fill_value=t)
        loss_adv = criterion_BCE(unk_class_p, unk_labels)
        train_loss_adv += loss_adv.item()
        loss_adv.backward()

        #重み更新
        optimizer_G.step()
        optimizer_C.step()

        # 進行状況バーを更新
        progress_bar.update(1)
        
    # エポック終了時に進行状況バーを閉じる
    progress_bar.close()

    #損失や出力を確認するための計算
    avg_train_loss_s = train_loss_s/batch_roop
    avg_train_loss_adv = train_loss_adv/batch_roop

    #評価
    model_G.eval()
    model_C.eval()

    with torch.no_grad():
        class_num_list = list(range(knw_class_num+1))
        avg_acc_C = {c:0 for c in class_num_list}
        avg_count_C = {c:0 for c in class_num_list}
        
        for images, labels in tqdm(loader_target):
            images = images.to(device)
            labels = labels.to(device)
            outputs_G = model_G(images)
            outputs_C = model_C(outputs_G)
    
            #予測値算出
            pred = outputs_C.argmax(dim=1)
            for cl in class_num_list:
                avg_acc_C[cl] += (pred[labels==cl] == labels[labels==cl]).float().sum()
                avg_count_C[cl] += labels[labels==cl].shape[0]  
        
        avg_C_knw = 0
        for cl in class_num_list:
            if avg_count_C[cl] == 0:
                avg_C_knw += 0
            else:
                if cl != knw_class_num:
                    avg_C_knw += (float(avg_acc_C[cl]) / float(avg_count_C[cl]))
                else:
                    avg_C_unk = (float(avg_acc_C[cl]) / float(avg_count_C[cl]))#Unk
        avg_C_knw /= float(knw_class_num)#OS*

        avg_C_knw *= 100#OS*
        avg_C_unk *= 100#Unk
        hos = (2*avg_C_knw*avg_C_unk)/(avg_C_knw+avg_C_unk)#HOS
        
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_s: {avg_train_loss_s:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_adv: {avg_train_loss_adv:.5f}")
        
        print(f'Epoch [{(i+1)}/{epoch}] OS*: {avg_C_knw:.2f} %')
        print(f'Epoch [{(i+1)}/{epoch}] Unk: {avg_C_unk:.2f} %')
        print(f'Epoch [{(i+1)}/{epoch}] HOS: {hos:.2f} ')
        
