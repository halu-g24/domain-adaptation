#論文の疑似コードと同じようにwとmをターゲットデータ数取得する
#ターゲットデータに数の多いAmazonで実行するとgpuメモリエラー発生した。
#↑を解消した。おそらく完成

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
source_path = './Office31/WtoD/source_images/'
target_path = './Office31/WtoD/target_images/'
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
    
class CustomDataset(Dataset):
    def __init__(self, image, label, weight, mask):
        self.image = image
        self.label = label
        self.weight = weight
        self.mask = mask
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.label[idx]
        weight = self.weight[idx]
        mask = self.mask[idx]
        return image, label, weight, mask
    

#特徴抽出器
class M(nn.Module):
    def __init__(self):
        super(M,self).__init__()
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*[x for x in list(resnet50.children())[:-1]]) # Upto the avgpool layer

    def forward(self,x):
        feats = self.features(x)
        return feats.view((x.shape[0], 2048))

class E(nn.Module):
    def __init__(self):
        super(E,self).__init__()
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
            nn.ELU()
        )
    def forward(self,x):
        feats = self.classifier(x)
        return feats
    
#プロトタイプベース分類器
class Gs(nn.Module):
    def __init__(self, initial_temperature=0.05):
        super(Gs,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, knw_class_num, bias=False)
        )
        self.temperature = initial_temperature

    def forward(self,x,reverse=False):
        if reverse:
            x = GradientReversalLayer.apply(x, 1)
        x = F.normalize(x)
        logits = self.classifier(x) / self.temperature
        return logits
    
class Gn(nn.Module):
    def __init__(self, initial_temperature=0.05):
        super(Gn,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, knw_class_num*4, bias=False)
        )
        self.temperature = initial_temperature

    def forward(self,x,reverse=False):
        if reverse:
            x = GradientReversalLayer.apply(x, 1)
        x = F.normalize(x)
        logits = self.classifier(x) / self.temperature
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

model_M = M().to(device)
model_E = E().to(device)
model_Gs = Gs().to(device)
model_Gn = Gn().to(device)

criterion_CE = nn.CrossEntropyLoss()

#エポック数
epoch = 600

#バッチサイズ
batch_size = 32

#学習率
lr = 0.00001

#PSDCのハイパーパラメータ
a = 0.1 #ガンマ
b = 0.2 #ラムダ0
c = 0.4 #ラムダ1
d = 0.4 #ラムダ2
k = 30 #疑似未知学習のターゲットデータ上位k%

#データローダー
dataset_source = TemplateDataset(source_path)
loader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
dataset_target = TemplateDataset(target_path)
loader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)

#オプティマイザー
optimizer_M = optim.SGD(model_M.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_E = optim.SGD(model_E.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_Gs = optim.SGD(model_Gs.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_Gn = optim.SGD(model_Gn.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

print(f'ソースデータ数: {len(dataset_source)}')
print(f'ターゲットデータ数: {len(dataset_target)}')
batch_roop = max(len(loader_source),len(loader_target))
print(f'バッチループ数: {batch_roop}')

for i in range(epoch):
    #学習
    model_M.train()
    model_E.train()
    model_Gs.train()
    model_Gn.train()

    #各loss
    train_loss_sid = 0
    train_loss_nod = 0
    train_loss_sh = 0
    train_loss_adv = 0
    train_loss_uk = 0

    #転移性指標と疑似未知学習のm
    all_target_images = []
    all_target_labels = []
    all_target_weight = []
    for target_images, target_labels in loader_target:
        all_target_images.append(target_images)
        all_target_labels.append(target_labels)
        with torch.no_grad():
            target_images = target_images.to(device)
            t_outputs_M = model_M(target_images)
            t_outputs_E = model_E(t_outputs_M)
            t_outputs_Gs = model_Gs(t_outputs_E)
            t_outputs_Gn = model_Gn(t_outputs_E)
            t_outputs_G = torch.cat([t_outputs_Gs, t_outputs_Gn], dim=-1)
            sm_t_outputs_G = F.softmax(t_outputs_G, dim=1)
            w, _ = torch.max(sm_t_outputs_G[:, :knw_class_num], dim=1)
            w = w.detach().cpu().numpy()
            all_target_weight.append(w)
            
    all_target_images = torch.cat(all_target_images)
    all_target_labels = torch.cat(all_target_labels)
    all_target_weight = np.concatenate(all_target_weight)
    
    sorted_w = np.argsort(all_target_weight)
    top_w_idx = sorted_w[:int(k/100 * len(sorted_w))]
    m = np.zeros(len(all_target_weight), dtype=int)
    m[top_w_idx] = 1
    w = torch.from_numpy(all_target_weight)
    m = torch.from_numpy(m)
    
    mask_target_dataset = CustomDataset(all_target_images, all_target_labels, w, m)
    mask_target_loader = DataLoader(mask_target_dataset, batch_size=batch_size, shuffle=True)

    #DAのミニバッチ処理
    source_iter = iter(loader_source)
    target_iter = iter(mask_target_loader)
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
            target_images, target_labels, w, m = next(target_iter)
            if target_images.shape[0] < batch_size:
                target_iter = iter(mask_target_loader)
                target_images, target_labels, w, m = next(target_iter)
        except StopIteration:
            target_iter = iter(mask_target_loader)
            target_images, target_labels, w, m = next(target_iter)
        
        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)
        target_labels = target_labels.to(device)
        w = w.to(device)
        m = m.to(device)

        #オープンセット認識
        optimizer_M.zero_grad()
        optimizer_E.zero_grad()
        optimizer_Gs.zero_grad()
        optimizer_Gn.zero_grad()

        #loss_sid
        s_outputs_M = model_M(source_images)
        s_outputs_E = model_E(s_outputs_M)
        s_outputs_Gs = model_Gs(s_outputs_E)
        s_outputs_Gn = model_Gn(s_outputs_E)
        s_outputs_G = torch.cat([s_outputs_Gs, s_outputs_Gn], dim=-1)
        loss_sid_1 = criterion_CE(s_outputs_G, source_labels)

        s_outputs_Gs_0 = torch.zeros(batch_size,knw_class_num).to(device)
        s_outputs_G_0 = torch.cat([s_outputs_Gs_0, s_outputs_Gn], dim=-1)
        y_ns = s_outputs_G_0.argmax(dim=1)
        loss_sid_2 = criterion_CE(s_outputs_G_0, y_ns)

        loss_sid = loss_sid_1 + a*loss_sid_2
        train_loss_sid += loss_sid.item()

        #特徴スプライシング
        E_x = s_outputs_E.detach().cpu().numpy()
        spliced = []
        for j in range(batch_size):
            u_n_a = E_x[j]
            dif_idx = torch.where(source_labels != source_labels[j])[0]
            random_idx = dif_idx[torch.randint(len(dif_idx), (1,)).item()]
            u_n_b = E_x[random_idx]
            sorted_u_n_a = np.flip(np.argsort(u_n_a))
            top_u_n_a_idx = sorted_u_n_a[:int(0.15 * len(sorted_u_n_a))]
            u_n_a[top_u_n_a_idx] = u_n_b[top_u_n_a_idx]
            spliced.append(u_n_a)
        
        spliced_feats = np.stack(spliced, axis=0)
        u_n = torch.from_numpy(spliced_feats).to(device)
         
        #loss_nod
        u_outputs_Gs = model_Gs(u_n)
        u_outputs_Gn = model_Gn(u_n)
        u_outputs_G = torch.cat([u_outputs_Gs, u_outputs_Gn], dim=-1)
        u_outputs_Gs_0 = torch.zeros(batch_size,knw_class_num).to(device)
        u_outputs_G_0 = torch.cat([u_outputs_Gs_0, u_outputs_Gn], dim=-1)
        y_nn = u_outputs_G_0.argmax(dim=1)
        loss_nod = criterion_CE(u_outputs_G, y_nn)
        train_loss_nod += loss_nod.item()

        loss = loss_sid + b*loss_nod

        #重み更新
        loss.backward()
        optimizer_M.step()
        optimizer_E.step()
        optimizer_Gs.step()
        optimizer_Gn.step()

        #重み付きアライメント
        optimizer_M.zero_grad()
        optimizer_E.zero_grad()
        optimizer_Gs.zero_grad()
        optimizer_Gn.zero_grad()

        #loss_sh
        t_outputs_M = model_M(target_images)
        t_outputs_E = model_E(t_outputs_M)
        t_outputs_Gs = model_Gs(t_outputs_E, reverse=True)
        sm_t_outputs_Gs = F.softmax(t_outputs_Gs, dim=1)
        h_Gs = -sm_t_outputs_Gs*torch.log(sm_t_outputs_Gs + 1e-8)
        h_Gs = h_Gs.sum(dim=1)
        loss_sh = (torch.sum(w*h_Gs))/len(h_Gs)
        train_loss_sh += loss_sh.item()

        loss = -loss_sh*c

        #重み更新
        loss.backward()
        optimizer_M.step()
        optimizer_E.step()
        optimizer_Gs.step()

        optimizer_M.zero_grad()
        optimizer_E.zero_grad()
        optimizer_Gs.zero_grad()
        optimizer_Gn.zero_grad()

        #loss_adv
        t_outputs_M = model_M(target_images)
        t_outputs_E = model_E(t_outputs_M)
        t_outputs_Gs = model_Gs(t_outputs_E, reverse=True)
        t_outputs_Gn = model_Gn(t_outputs_E, reverse=True)
        t_outputs_G = torch.cat([t_outputs_Gs, t_outputs_Gn], dim=-1)
        sm_t_outputs_G = F.softmax(t_outputs_G, dim=1)
        s_ = torch.sum(sm_t_outputs_G[:, :knw_class_num], dim=1)
        loss_adv = -w*torch.log(1-s_ + 1e-8)-(1-w)*torch.log(s_ + 1e-8)
        loss_adv = (torch.sum(loss_adv))/len(loss_adv)
        train_loss_adv += loss_adv.item()
        
        loss = loss_adv*d

        #重み更新
        loss.backward()
        optimizer_M.step()
        optimizer_E.step()
        optimizer_Gs.step()
        optimizer_Gn.step()

        #疑似未知学習
        optimizer_M.zero_grad()
        optimizer_E.zero_grad()
        optimizer_Gs.zero_grad()
        optimizer_Gn.zero_grad()

        #loss_uk最大化
        t_outputs_M = model_M(target_images)
        t_outputs_E = model_E(t_outputs_M)
        t_outputs_Gs = model_Gs(t_outputs_E)
        sm_t_outputs_Gs = F.softmax(t_outputs_Gs, dim=1)
        h_Gs = -sm_t_outputs_Gs*torch.log(sm_t_outputs_Gs + 1e-8)
        h_Gs = h_Gs.sum(dim=1)
        loss_uk = (torch.sum(h_Gs*m))/len(m[m==1])
        train_loss_uk += loss_uk.item()

        loss = -loss_uk

        #重み更新
        loss.backward()
        optimizer_M.step()
        optimizer_E.step()
        optimizer_Gs.step()

        # 進行状況バーを更新
        progress_bar.update(1)
        
    # エポック終了時に進行状況バーを閉じる
    progress_bar.close()

    #損失や出力を確認するための計算
    avg_train_loss_sid = train_loss_sid/batch_roop
    avg_train_loss_nod = train_loss_nod/batch_roop*b
    avg_train_loss_sh = train_loss_sh/batch_roop*c
    avg_train_loss_adv = train_loss_adv/batch_roop*d
    avg_train_loss_uk = train_loss_uk/batch_roop

    #評価
    model_M.eval()
    model_E.eval()
    model_Gs.eval()
    model_Gn.eval()

    with torch.no_grad():
        class_num_list = list(range(knw_class_num+1))
        avg_acc_C = {c:0 for c in class_num_list}
        avg_count_C = {c:0 for c in class_num_list}
        
        for images, labels in tqdm(loader_target):
            images = images.to(device)
            labels = labels.to(device)
            outputs_M = model_M(images)
            outputs_E = model_E(outputs_M)
            outputs_Gs = model_Gs(outputs_E)
            outputs_Gn = model_Gn(outputs_E)
            outputs_G = torch.cat([outputs_Gs, outputs_Gn], dim=-1)
    
            #予測値算出
            pred = outputs_G.argmax(dim=1)
            pred = torch.clamp(pred, max=knw_class_num)
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
        
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_sid: {avg_train_loss_sid:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_nod: {avg_train_loss_nod:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_sh: {avg_train_loss_sh:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_adv: {avg_train_loss_adv:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_uk: {avg_train_loss_uk:.5f}")

        print(f'Epoch [{(i+1)}/{epoch}] OS*: {avg_C_knw:.2f} %')
        print(f'Epoch [{(i+1)}/{epoch}] Unk: {avg_C_unk:.2f} %')
        print(f'Epoch [{(i+1)}/{epoch}] HOS: {hos:.2f} ')
        