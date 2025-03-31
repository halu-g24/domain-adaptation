#提案手法

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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

#Office31
source_path = './Office31/AtoD/source_images/'
target_path = './Office31/AtoD/target_images/'
knw_class_num = 10
unk_class_num = 1

#OfficeHome
#source_path = './OfficeHome/ArtoRw/source_images/'
#target_path = './OfficeHome/ArtoRw/target_images/'
#knw_class_num = 25
#unk_class_num = 6

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
    
#分類器
class C_knw(nn.Module):
    def __init__(self):
        super(C_knw,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, knw_class_num)
        )
    def forward(self,x):
        logits = self.classifier(x)
        return logits
    
class C_unk(nn.Module):
    def __init__(self):
        super(C_unk,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, knw_class_num*unk_class_num)
        )
    def forward(self,x):
        logits = self.classifier(x)
        return logits

#gpu check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_F = M().to(device)
model_E = E().to(device)
model_C_knw = C_knw().to(device)
model_C_unk = C_unk().to(device)

criterion_CE = nn.CrossEntropyLoss()
criterion_BCE = nn.BCELoss()

#エポック数
epoch = 200

#バッチサイズ
batch_size = 32

#学習率
lr = 0.0001

#ハイパーパラメータ
t = 0.5

#最大値取得
max_hos = 0
then_knw = 0
then_unk = 0
epoch_hos = 0
max_w_knw = 0
epoch_w_knw = 0
min_w_unk = 1
epoch_w_unk = 0

#データローダー
dataset_source = TemplateDataset(source_path)
loader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
dataset_target = TemplateDataset(target_path)
loader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)

#オプティマイザー
optimizer_F = optim.SGD(model_F.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_E = optim.SGD(model_E.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_C_knw = optim.SGD(model_C_knw.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_C_unk = optim.SGD(model_C_unk.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

print(f"実験: {source_path.split('/')[2]}")
print(f'ソースデータ数: {len(dataset_source)}')
print(f'ターゲットデータ数: {len(dataset_target)}')
batch_roop = max(len(loader_source),len(loader_target))
print(f'バッチループ数: {batch_roop}')

#可視化
interval = 5
dir = 'figures/my_method/AtoD/256/241202/method1'
array_s_ce = []
array_t_bce_max = []
array_t_bce_min = []
array_t_e = []
array_w_knw = []
array_w_unk = []
array_sum_knw_knw = []
array_sum_knw_unk = []
array_knw = []
array_unk = []
array_hos = []

for i in range(epoch):
    #学習
    model_F.train()
    model_E.train()
    model_C_knw.train()
    model_C_unk.train()

    #各loss
    train_loss_s_ce = 0
    train_loss_t_bce_max = 0
    train_loss_t_bce_min = 0
    train_loss_t_e = 0
    
    #特徴ベクトル取得用
    all_fs = np.empty((0, 256))
    all_ft = np.empty((0, 256))
    #正解ラベル取得用
    all_fs_label = np.empty((0))
    all_ft_label = np.empty((0))
    #重み取得用
    all_target_labels = np.empty((0))
    all_target_weight = np.empty((0))
    all_target_sum_knw = np.empty((0))

   
    #DAのミニバッチ処理
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
        
        target_labels_numpy = target_labels.detach().cpu().numpy()
        all_target_labels = np.concatenate((all_target_labels,target_labels_numpy), axis=0)
        
        #ソースでクロスエントロピー
        #loss_s_ce
        optimizer_F.zero_grad()
        optimizer_E.zero_grad()
        optimizer_C_knw.zero_grad()
        optimizer_C_unk.zero_grad()
        
        s_outputs_F = model_F(source_images)
        s_outputs_E = model_E(s_outputs_F)
        ###特徴ベクトル可視化
        #ソース
        fs = s_outputs_E.detach().cpu().numpy()
        all_fs = np.concatenate((all_fs,fs), axis=0)
        ls = torch.where(source_labels < knw_class_num, 0,1)
        ls = ls.detach().cpu().numpy()
        all_fs_label = np.concatenate((all_fs_label,ls), axis=0)
        #ターゲット
        with torch.no_grad():
            t_outputs_F = model_F(target_images)
            t_outputs_E = model_E(t_outputs_F)
        ft = t_outputs_E.detach().cpu().numpy()
        all_ft = np.concatenate((all_ft,ft), axis=0)
        lt = torch.where(target_labels < knw_class_num, 1,2)
        lt = lt.detach().cpu().numpy()
        all_ft_label = np.concatenate((all_ft_label,lt), axis=0)
        ###
        s_outputs_C_knw = model_C_knw(s_outputs_E)
        s_outputs_C_unk = model_C_unk(s_outputs_E)
        s_outputs_C = torch.cat([s_outputs_C_knw, s_outputs_C_unk], dim=-1)
        loss_s_ce = criterion_CE(s_outputs_C, source_labels)
        train_loss_s_ce += loss_s_ce.item()

        loss = loss_s_ce
         
        #重み更新
        loss.backward()
        optimizer_F.step()
        optimizer_E.step()
        optimizer_C_knw.step()
        optimizer_C_unk.step()


        #全ターゲットを弱く未知とする。分類器のみ重み更新。
        #loss_t_bce_min
        optimizer_F.zero_grad()
        optimizer_E.zero_grad()
        optimizer_C_knw.zero_grad()
        optimizer_C_unk.zero_grad()
        
        t_outputs_F = model_F(target_images)
        t_outputs_E = model_E(t_outputs_F)
        t_outputs_C_knw = model_C_knw(t_outputs_E)
        t_outputs_C_unk = model_C_unk(t_outputs_E)
        t_outputs_C = torch.cat([t_outputs_C_knw, t_outputs_C_unk], dim=-1)
        sm_t_outputs_C = F.softmax(t_outputs_C, dim=1)
        sum_knw_t_outputs_C = torch.sum(sm_t_outputs_C[:, :knw_class_num], dim=1)
        t_labels = torch.full_like(sum_knw_t_outputs_C, fill_value=t)
        #予測値をクリップ
        epsilon = 1e-7
        clamp_outputs = torch.clamp(sum_knw_t_outputs_C, epsilon, 1 - epsilon)
        loss_t_bce_min = criterion_BCE(clamp_outputs,t_labels)
        train_loss_t_bce_min += loss_t_bce_min.item()

        loss = loss_t_bce_min
         
        #重み更新
        loss.backward()
        optimizer_C_knw.step()
        optimizer_C_unk.step()

        
        #既知と未知の類似度重みを引き離す。特徴抽出器のみ重み更新。
        #loss_t_bce_max
        optimizer_F.zero_grad()
        optimizer_E.zero_grad()
        optimizer_C_knw.zero_grad()
        optimizer_C_unk.zero_grad()
        
        t_outputs_F = model_F(target_images)
        t_outputs_E = model_E(t_outputs_F)
        t_outputs_C_knw = model_C_knw(t_outputs_E)
        t_outputs_C_unk = model_C_unk(t_outputs_E)
        t_outputs_C = torch.cat([t_outputs_C_knw, t_outputs_C_unk], dim=-1)
        sm_t_outputs_C = F.softmax(t_outputs_C, dim=1)
        sum_knw_t_outputs_C = torch.sum(sm_t_outputs_C[:, :knw_class_num], dim=1)
        t_labels = torch.full_like(sum_knw_t_outputs_C, fill_value=t)
        with torch.no_grad():
            w, _ = torch.max(sm_t_outputs_C[:, :knw_class_num], dim=1)
            w_numpy = w.detach().cpu().numpy()
            all_target_weight = np.concatenate((all_target_weight,w_numpy), axis=0)
            sum_knw = torch.sum(sm_t_outputs_C[:, :knw_class_num], dim=1)
            sum_knw_numpy = sum_knw.detach().cpu().numpy()
            all_target_sum_knw = np.concatenate((all_target_sum_knw,sum_knw_numpy), axis=0)
        #予測値をクリップ
        epsilon = 1e-7
        clamp_outputs = torch.clamp(2*t_labels - sum_knw_t_outputs_C, epsilon, 1 - epsilon)
        loss_t_bce_max = criterion_BCE(clamp_outputs,w)
        train_loss_t_bce_max += loss_t_bce_max.item()

        loss = -loss_t_bce_max
         
        #重み更新
        loss.backward()
        optimizer_F.step()
        optimizer_E.step()
        
        
        


        # 進行状況バーを更新
        progress_bar.update(1)
        
    # エポック終了時に進行状況バーを閉じる
    progress_bar.close()

    #損失や出力を確認するための計算
    avg_train_loss_s_ce = train_loss_s_ce/batch_roop
    avg_train_loss_t_bce_min = train_loss_t_bce_min/batch_roop
    avg_train_loss_t_bce_max = train_loss_t_bce_max/batch_roop

    mask_knw = (all_target_labels < knw_class_num)
    mask_unk = (all_target_labels == knw_class_num)
    mean_w_knw = all_target_weight[mask_knw].mean().item()
    mean_w_unk = all_target_weight[mask_unk].mean().item()
    mean_sum_knw_knw = all_target_sum_knw[mask_knw].mean().item()
    mean_sum_knw_unk = all_target_sum_knw[mask_unk].mean().item()
    print(mean_sum_knw_knw)
    print(mean_w_knw)
    print(mean_sum_knw_unk)
    print(mean_w_unk)

    #評価
    model_F.eval()
    model_E.eval()
    model_C_knw.eval()
    model_C_unk.eval()

    with torch.no_grad():
        class_num_list = list(range(knw_class_num+1))
        avg_acc_C = {c:0 for c in class_num_list}
        avg_count_C = {c:0 for c in class_num_list}
        
        for images, labels in tqdm(loader_target):
            images = images.to(device)
            labels = labels.to(device)
            outputs_F = model_F(images)
            outputs_E = model_E(outputs_F)
            outputs_C_knw = model_C_knw(outputs_E)
            outputs_C_unk = model_C_unk(outputs_E)
            outputs_C = torch.cat([outputs_C_knw, outputs_C_unk], dim=-1)
    
            #予測値算出
            pred = outputs_C.argmax(dim=1)
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
        
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_s_ce: {avg_train_loss_s_ce:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_t_bce_min: {avg_train_loss_t_bce_min:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_t_bce_max: {avg_train_loss_t_bce_max:.5f}")
        
        print(f'Epoch [{(i+1)}/{epoch}] OS*: {avg_C_knw:.2f} %')
        print(f'Epoch [{(i+1)}/{epoch}] Unk: {avg_C_unk:.2f} %')
        print(f'Epoch [{(i+1)}/{epoch}] HOS: {hos:.2f} ')

        if max_hos <= hos:
            max_hos = hos
            then_knw = avg_C_knw
            then_unk = avg_C_unk
            epoch_hos = i+1
        
        if max_w_knw <= mean_w_knw:
            max_w_knw = mean_w_knw
            epoch_w_knw = i+1

        if min_w_unk >= mean_w_unk:
            min_w_unk = mean_w_unk
            epoch_w_unk = i+1


        if (i+1)%interval == 0:
            array_s_ce.append(round(avg_train_loss_s_ce,4))
            array_t_bce_min.append(round(avg_train_loss_t_bce_min,4))
            array_t_bce_max.append(round(avg_train_loss_t_bce_max,4))
            array_w_knw.append(round(mean_w_knw,4))
            array_w_unk.append(round(mean_w_unk,4))
            array_sum_knw_knw.append(round(mean_sum_knw_knw,4))
            array_sum_knw_unk.append(round(mean_sum_knw_unk,4))
            array_knw.append(round(avg_C_knw,2))
            array_unk.append(round(avg_C_unk,2))
            array_hos.append(round(hos,2))

        ###特徴ベクトル可視化
        #sourceとtargetを結合
        all_f = np.concatenate((all_fs,all_ft), axis=0)
        all_f_label = np.concatenate((all_fs_label,all_ft_label), axis=0)
        if (i+1)%(interval*4) == 0 or i == 0:
            #2次元可視化
            print('t-SNE')
            f_tsne = TSNE(n_components=2, random_state=0).fit_transform(all_f)
            #colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'yellow']
            colors = ['blue', 'red', 'green']
            cmap = ListedColormap(colors)
            plt.scatter(f_tsne[:,0], f_tsne[:,1], c=all_f_label, cmap=cmap, s=5)
            plt.colorbar(ticks=range(len(colors)))
            plt.title(f'{i+1}epoch OS*: {avg_C_knw:.2f} Unk: {avg_C_unk:.2f} HOS: {hos:.2f}', loc='center')
            file_path = os.path.join(dir, f'dim_red_{i+1}epoch.png')  # フォルダパスとファイル名を結合
            plt.savefig(file_path, dpi=300, bbox_inches='tight')  # 高解像度で保存
            plt.close()
        ###

#最大値表示
print(f'Epoch [{epoch_hos}/{epoch}], max_HOS: {max_hos:.2f} then_OS*: {then_knw:.2f} then_Unk: {then_unk:.2f}')


x = range(0,epoch,interval)

#HOS,OS*,UNK
plt.plot(x, array_knw, marker='.', label='OS*')
plt.plot(x, array_unk, marker='.', label='Unk')
plt.plot(x, array_hos, marker='.', label='HOS')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc(%)')
plt.ylim(0,105)
plt.title(f'Epoch [{epoch_hos}/{epoch}], max_HOS: {max_hos:.2f} OS*: {then_knw:.2f} Unk: {then_unk:.2f}', loc='center')
file_path = os.path.join(dir, f'hos.png')  # フォルダパスとファイル名を結合
plt.savefig(file_path, dpi=300, bbox_inches='tight')  # 高解像度で保存
plt.close()

#Loss
plt.plot(x, array_s_ce, marker='.', label='l_s_ce')
plt.plot(x, array_t_bce_min, marker='.', label='l_t_bce_min')
plt.plot(x, array_t_bce_max, marker='.', label='l_t_bce_max')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title("Loss", loc='center')
file_path = os.path.join(dir, f'loss.png')  # フォルダパスとファイル名を結合
plt.savefig(file_path, dpi=300, bbox_inches='tight')  # 高解像度で保存
plt.close()

#w
plt.plot(x, array_w_knw, marker='.', label='w_knw')
plt.plot(x, array_w_unk, marker='.', label='w_unk')
plt.plot(x, array_sum_knw_knw, marker='.', label='sum_knw_knw')
plt.plot(x, array_sum_knw_unk, marker='.', label='sum_knw_unk')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('w')
plt.title(f'Epoch [{epoch_w_knw}/{epoch}], max_w_knw: {max_w_knw:.2f} Epoch [{epoch_w_unk}/{epoch}], min_w_unk: {min_w_unk:.2f}', loc='center')
file_path = os.path.join(dir, f'w.png')  # フォルダパスとファイル名を結合
plt.savefig(file_path, dpi=300, bbox_inches='tight')  # 高解像度で保存
plt.close()
