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
source_path = './Office31_exp/DtoA_CS/source_images/'
target_path = './Office31_exp/DtoA_CS/target_images/'
knw_class_num = 10

#OfficeHome
#source_path = './OfficeHome_exp/RwtoAr_CS/source_images/'
#target_path = './OfficeHome_exp/RwtoAr_CS/target_images/'
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
            nn.Linear(256,knw_class_num)
        )
    def forward(self,x):
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

max_accs = []
max_idxs = []

for x in range(3):
    model_F = M().to(device)
    model_C1 = C().to(device)
    model_C2 = C().to(device)

    criterion_CE = nn.CrossEntropyLoss()
    criterion_L1 = nn.L1Loss()

    #エポック数
    epoch = 200

    #バッチサイズ
    batch_size = 32

    #学習率
    lr = 0.0001

    #MCDのハイパーパラメータ
    n = 4

    #データローダー
    dataset_source = TemplateDataset(source_path)
    loader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
    dataset_target = TemplateDataset(target_path)
    loader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)

    #オプティマイザー
    optimizer_F = optim.SGD(model_F.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
    optimizer_C1 = optim.SGD(model_C1.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
    optimizer_C2 = optim.SGD(model_C2.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

    print(f'ソースデータ数: {len(dataset_source)}')
    print(f'ターゲットデータ数: {len(dataset_target)}')
    batch_roop = max(len(loader_source),len(loader_target))
    print(f'バッチループ数: {batch_roop}')

    #最大値取得
    max_acc = 0
    max_idx = 0

    for i in range(epoch):
        #学習
        model_F.train()
        model_C1.train()
        model_C2.train()

        #各loss
        train_loss_ce = 0
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


            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            target_images = target_images.to(device)
            target_labels = target_labels.to(device)

            #stepA
            optimizer_F.zero_grad()
            optimizer_C1.zero_grad()
            optimizer_C2.zero_grad()
            s_outputs_F = model_F(source_images)
            s_outputs_C1 = model_C1(s_outputs_F)
            s_outputs_C2 = model_C2(s_outputs_F)
            loss_ce1 = criterion_CE(s_outputs_C1, source_labels)
            loss_ce2 = criterion_CE(s_outputs_C2, source_labels)
            loss_ce = loss_ce1 + loss_ce2
            train_loss_ce += loss_ce.item()
            loss_ce.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()

            
            #stepB
            optimizer_C1.zero_grad()
            optimizer_C2.zero_grad()
            s_outputs_F = model_F(source_images)
            s_outputs_C1 = model_C1(s_outputs_F)
            s_outputs_C2 = model_C2(s_outputs_F)
            loss_ce1 = criterion_CE(s_outputs_C1, source_labels)
            loss_ce2 = criterion_CE(s_outputs_C2, source_labels)
            loss_ce = loss_ce1 + loss_ce2

            t_outputs_F = model_F(target_images)
            t_outputs_C1 = model_C1(t_outputs_F)
            t_outputs_C2 = model_C2(t_outputs_F)
            t_sm_outputs_C1 = F.softmax(t_outputs_C1, dim=1)
            t_sm_outputs_C2 = F.softmax(t_outputs_C2, dim=1)
            loss_adv = criterion_L1(t_sm_outputs_C1,t_sm_outputs_C2)
            train_loss_adv += loss_adv.item()
            loss = loss_ce - loss_adv
            loss.backward()
            optimizer_C1.step()
            optimizer_C2.step()
            
            #stepC
            for j in range(n):
                optimizer_F.zero_grad()
                t_outputs_F = model_F(target_images)
                t_outputs_C1 = model_C1(t_outputs_F)
                t_outputs_C2 = model_C2(t_outputs_F)
                t_sm_outputs_C1 = F.softmax(t_outputs_C1, dim=1)
                t_sm_outputs_C2 = F.softmax(t_outputs_C2, dim=1)
                loss_adv = criterion_L1(t_sm_outputs_C1,t_sm_outputs_C2)
                loss_adv.backward()
                optimizer_F.step()

            # 進行状況バーを更新
            progress_bar.update(1)
            
        # エポック終了時に進行状況バーを閉じる
        progress_bar.close()

        #損失や出力を確認するための計算
        avg_train_loss_ce = train_loss_ce/batch_roop
        avg_train_loss_adv = train_loss_adv/batch_roop
        
        #評価
        model_F.eval()
        model_C1.eval()
        model_C2.eval()

        with torch.no_grad():
            class_num_list = list(range(knw_class_num))
            avg_acc_C = {c:0 for c in class_num_list}
            avg_count_C = {c:0 for c in class_num_list}
            
            for images, labels in tqdm(loader_target):
                images = images.to(device)
                labels = labels.to(device)
                outputs_F = model_F(images)
                outputs_C = model_C1(outputs_F)
        
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
                    avg_C_knw += (float(avg_acc_C[cl]) / float(avg_count_C[cl]))
                    
            avg_C_knw /= float(knw_class_num)#OS*

            avg_C_knw *= 100#OS*
            
            
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_ce: {avg_train_loss_ce:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_adv: {avg_train_loss_adv:.5f}")
        
        print(f'Epoch [{(i+1)}/{epoch}] OS*: {avg_C_knw:.2f} %')
        
        if max_acc <= avg_C_knw:
            max_acc = avg_C_knw
            max_idx = i+1
        
    #1試行の最大値
    print(f'{x+1}試行目 Epoch [{max_idx}/{epoch}], max_OS*: {max_acc:.2f} ')
    
    max_accs.append(round(max_acc,2))
    max_idxs.append(max_idx)

#3試行の平均と標準偏差
print(f'max_accs:{max_accs} idxs:{max_idxs} mean:{round(np.mean(max_accs),2)} std:{round(np.std(max_accs),2)}')
