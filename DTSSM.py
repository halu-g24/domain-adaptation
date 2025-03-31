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
            nn.Linear(256,knw_class_num)
        )
    def forward(self,x):
        logits = self.classifier(x)
        return logits

#最終出力分類器
class C_(nn.Module):
    def __init__(self):
        super(C_,self).__init__()
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
    
#識別器
class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha):
        x = GradientReversalLayer.apply(x, alpha)
        x = self.discriminator(x)
        return x
    
#gpu check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_F = M().to(device)
model_Cta = C().to(device)
model_Ctb = C().to(device)
model_Csa = C().to(device)
model_Csb = C().to(device)
model_D = D().to(device)
model_C = C_().to(device)

criterion_CE = nn.CrossEntropyLoss()
criterion_L1 = nn.L1Loss()

#エポック数
epoch = 300

#バッチサイズ
batch_size = 32

#学習率
lr = 0.0002

#学習率減衰
def lr_decay(ep):
    p = ep/(epoch-1)
    lr_d = 1/((1+10*p)**0.75)
    return lr_d

#DTSSMのハイパーパラメータ
r = 0.5
a = 0.15
b = 0.25
c = 0.1
d = 0.01

#データローダー
dataset_source = TemplateDataset(source_path)
loader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
dataset_target = TemplateDataset(target_path)
loader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)

#オプティマイザー
optimizer_F = optim.SGD(model_F.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_Cta = optim.SGD(model_Cta.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_Ctb = optim.SGD(model_Ctb.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_Csa = optim.SGD(model_Csa.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_Csb = optim.SGD(model_Csb.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_D = optim.SGD(model_D.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
optimizer_C = optim.SGD(model_C.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

#学習率スケジューラー
scheduler_F = optim.lr_scheduler.LambdaLR(optimizer_F, lr_decay)
scheduler_Cta = optim.lr_scheduler.LambdaLR(optimizer_Cta, lr_decay)
scheduler_Ctb = optim.lr_scheduler.LambdaLR(optimizer_Ctb, lr_decay)
scheduler_Csa = optim.lr_scheduler.LambdaLR(optimizer_Csa, lr_decay)
scheduler_Csb = optim.lr_scheduler.LambdaLR(optimizer_Csb, lr_decay)
scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_decay)
scheduler_C = optim.lr_scheduler.LambdaLR(optimizer_C, lr_decay)

print(f'ソースデータ数: {len(dataset_source)}')
print(f'ターゲットデータ数: {len(dataset_target)}')
batch_roop = max(len(loader_source),len(loader_target))
print(f'バッチループ数: {batch_roop}')

for i in range(epoch):
    #学習
    model_F.train()
    model_Cta.train()
    model_Ctb.train()
    model_Csa.train()
    model_Csb.train()
    model_D.train()
    model_C.train()

    #勾配反転層パラメータ
    p = i/(epoch-1)
    lambda_ = 2/(1+math.exp(-10*p))-1

    #各loss
    train_loss_tce = 0
    train_loss_im = 0
    train_loss_pd = 0
    train_loss_te = 0
    train_loss_d = 0
    train_loss_t = 0
    train_loss_s = 0
    train_loss_e = 0

    #unkが1以下の場合の処理用
    count_loss_t = 0

    #Dの出力確認
    train_s_output_d = 0
    train_t_output_d = 0

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

        optimizer_F.zero_grad()
        optimizer_Cta.zero_grad()
        optimizer_Ctb.zero_grad()
        optimizer_Csa.zero_grad()
        optimizer_Csb.zero_grad()
        optimizer_D.zero_grad()
        optimizer_C.zero_grad()

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)
        target_labels = target_labels.to(device)

        #Step1
        #loss_tce
        s_outputs_F = model_F(source_images)
        s_outputs_Cta = model_Cta(s_outputs_F)
        s_outputs_Ctb = model_Ctb(s_outputs_F)
        loss_tce = criterion_CE(s_outputs_Cta, source_labels) + criterion_CE(s_outputs_Ctb, source_labels)
        train_loss_tce += loss_tce.item()

        #loss_im
        s_outputs_Csa = model_Csa(s_outputs_F)
        s_outputs_Csb = model_Csb(s_outputs_F)
        loss_im = criterion_L1(s_outputs_Csa, s_outputs_Cta) + criterion_L1(s_outputs_Csb, s_outputs_Ctb)
        train_loss_im += loss_im.item()

        #loss_pd
        #最大化
        loss_pd = criterion_L1(s_outputs_Cta, s_outputs_Ctb)
        train_loss_pd += loss_pd.item()

        #類似度スコア計算
        with torch.no_grad():
            t_outputs_F = model_F(target_images)
            t_outputs_Csa = model_Csa(t_outputs_F)
            t_outputs_Csb = model_Csb(t_outputs_F)
            sm_t_outputs_Csa = F.softmax(t_outputs_Csa, dim=1)
            sm_t_outputs_Csb = F.softmax(t_outputs_Csb, dim=1)
            max_Csa, posi_Csa = torch.max(sm_t_outputs_Csa, dim=1)#最大確率値と位置
            max_Csb, posi_Csb = torch.max(sm_t_outputs_Csb, dim=1)#最大確率値と位置

            sc =(max_Csa + max_Csb)/2#類似度スコア
            w = (sc - torch.min(sc))/(torch.max(sc) - torch.min(sc))#類似度重み
            sm = F.softmax(w, dim=0)#類似度重みをsoftmaxしたもの sm(w)
            sm_ = F.softmax(1-w, dim=0)#類似度重みを1から引いてsoftmaxしたもの sm(1-w)
            
            #既知と未知の分離
            knw_posi = (w > r).nonzero().squeeze()
            unk_posi = (w < r).nonzero().squeeze()
            p1_knw_knw_posi = knw_posi[target_labels[knw_posi]!=knw_class_num]
            p1_knw_unk_posi = knw_posi[target_labels[knw_posi]==knw_class_num]
            p1_unk_knw_posi = unk_posi[target_labels[unk_posi]!=knw_class_num]
            p1_unk_unk_posi = unk_posi[target_labels[unk_posi]==knw_class_num]

            #類似度スコアの可視化より細かく
            same_posi = (posi_Csa == posi_Csb).nonzero()
            dif_posi = (posi_Csa != posi_Csb).nonzero()
            same_knw_posi = same_posi[w[same_posi] > r]#2人の生徒の推論クラスが一致　かつ　0.5より大きい類似度重みの位置を取得
            same_unk_posi = same_posi[w[same_posi] < r]#2人の生徒の推論クラスが一致　かつ　0.5より小さい類似度重みの位置を取得
            dif_knw_posi = dif_posi[w[dif_posi] > r]#2人の生徒の推論クラスが一致しない　かつ　0.5より大きい類似度重みの位置を取得
            dif_unk_posi = dif_posi[w[dif_posi] < r]#2人の生徒の推論クラスが一致しない　かつ　0.5より小さい類似度重みの位置を取得
            p2_knw_knw_posi = same_knw_posi[target_labels[same_knw_posi]!=knw_class_num]
            p2_knw_unk_posi = same_knw_posi[target_labels[same_knw_posi]==knw_class_num]
            p2_unk_knw_posi = same_unk_posi[target_labels[same_unk_posi]!=knw_class_num]
            p2_unk_unk_posi = same_unk_posi[target_labels[same_unk_posi]==knw_class_num]
            p2_dif_knw_posi = dif_posi[target_labels[dif_posi]!=knw_class_num]
            p2_dif_unk_posi = dif_posi[target_labels[dif_posi]==knw_class_num]


        #loss_te
        t_outputs_F = model_F(target_images)
        t_outputs_Cta = model_Cta(t_outputs_F)
        t_outputs_Ctb = model_Ctb(t_outputs_F)
        sm_t_outputs_Cta = F.softmax(t_outputs_Cta, dim=1)
        sm_t_outputs_Ctb = F.softmax(t_outputs_Ctb, dim=1)

        h_Cta = -sm_t_outputs_Cta * torch.log(sm_t_outputs_Cta + 1e-8)
        h_Cta = h_Cta.sum(dim=1)
        h_Ctb = -sm_t_outputs_Ctb * torch.log(sm_t_outputs_Ctb + 1e-8)
        h_Ctb = h_Ctb.sum(dim=1)
        loss_te_a = torch.sum(sm*h_Cta) - torch.sum(sm_*h_Cta)
        loss_te_b = torch.sum(sm*h_Ctb) - torch.sum(sm_*h_Ctb)
        loss_te = loss_te_a + loss_te_b
        train_loss_te +=loss_te.item()

        #Step2
        #loss_d
        s_outputs_D = model_D(s_outputs_F, lambda_).squeeze()#1にしたい
        t_outputs_D = model_D(t_outputs_F, lambda_).squeeze()#0にしたい
        #正解ラベル作成
        one_labels = torch.ones_like(s_outputs_D)
        zero_labels = torch.zeros_like(t_outputs_D)

        criterion_Ld = nn.BCELoss()
        loss_d_one = criterion_Ld(s_outputs_D, one_labels)
        criterion_sm_Ld = nn.BCELoss(weight=sm,reduction= 'sum')#ソフトマックスした類似度重みを設定
        loss_d_zero = criterion_sm_Ld(t_outputs_D, zero_labels)
        loss_d = loss_d_one + loss_d_zero
        train_loss_d += loss_d.item()

        #出力D確認
        train_s_output_d += torch.mean(s_outputs_D)
        train_t_output_d += torch.mean(t_outputs_D)

        #loss_t
        if len(same_unk_posi) > 1:
            unk_t_outputs_F = t_outputs_F[same_unk_posi]
            unk_t_outputs_C = model_C(unk_t_outputs_F)
            unk_labels = torch.full_like(unk_t_outputs_C.argmax(dim=1), fill_value=knw_class_num)#未知ターゲットであろうもののラベル作り
            loss_t = criterion_CE(unk_t_outputs_C, unk_labels)
            train_loss_t += loss_t.item()
            count_loss_t += 1
        else:
            loss_t = 0
        
        #loss_s
        s_outputs_C = model_C(s_outputs_F)
        loss_s = criterion_CE(s_outputs_C, source_labels)
        train_loss_s += loss_s.item()

        #loss_e
        t_outputs_C = model_C(t_outputs_F)
        sm_t_outputs_C = F.softmax(t_outputs_C, dim=1)
        h_C = -sm_t_outputs_C * torch.log(sm_t_outputs_C + 1e-8)
        h_C = h_C.sum(dim=1)
        loss_e = (torch.sum(h_C))/len(h_C)
        train_loss_e += loss_e.item()


        #DTSSMの全損失
        loss = loss_tce + loss_im - d*loss_pd + c*loss_te + loss_d + a*loss_t + loss_s + b*loss_e

        #重み更新
        loss.backward()
        optimizer_F.step()
        optimizer_Cta.step()
        optimizer_Ctb.step()
        optimizer_Csa.step()
        optimizer_Csb.step()
        optimizer_D.step()
        optimizer_C.step()

        # 進行状況バーを更新
        progress_bar.update(1)
        
    # エポック終了時に進行状況バーを閉じる
    progress_bar.close()

    #損失や出力を確認するための計算
    avg_train_loss_tce = train_loss_tce/batch_roop
    avg_train_loss_im = train_loss_im/batch_roop
    avg_train_loss_pd = train_loss_pd/batch_roop*d
    avg_train_loss_te = train_loss_te/batch_roop*c
    avg_train_loss_d = train_loss_d/batch_roop
    avg_train_loss_t = train_loss_t/count_loss_t*a
    avg_train_loss_s = train_loss_s/batch_roop
    avg_train_loss_e = train_loss_e/batch_roop*b

    avg_train_s_output_d = train_s_output_d/batch_roop
    avg_train_t_output_d = train_t_output_d/batch_roop

    #学習率の減衰の確認
    current_lr_F = optimizer_F.param_groups[0]['lr']
    #print(current_lr_F)

    #学習率の減衰
    scheduler_F.step()
    scheduler_Cta.step()
    scheduler_Ctb.step()
    scheduler_Csa.step()
    scheduler_Csb.step()
    scheduler_D.step()
    scheduler_C.step()

    #評価
    model_F.eval()
    model_Cta.eval()
    model_Ctb.eval()
    model_Csa.eval()
    model_Csb.eval()
    model_D.eval()
    model_C.eval()

    with torch.no_grad():
        class_num_list = list(range(knw_class_num+1))
        avg_acc_C = {c:0 for c in class_num_list}
        avg_count_C = {c:0 for c in class_num_list}
        
        for images, labels in tqdm(loader_target):
            images = images.to(device)
            labels = labels.to(device)
            outputs_F = model_F(images)
            outputs_C = model_C(outputs_F)
    
            #予測値算出
            pred = outputs_C.max(1)[1]
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
        
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_tce: {avg_train_loss_tce:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_im: {avg_train_loss_im:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_pd: {avg_train_loss_pd:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_te: {avg_train_loss_te:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_d: {avg_train_loss_d:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_t: {avg_train_loss_t:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_s: {avg_train_loss_s:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_loss_e: {avg_train_loss_e:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_s_output_d: {avg_train_s_output_d:.5f}")
        print(f"Epoch [{(i+1)}/{epoch}] avg_train_t_output_d: {avg_train_t_output_d:.5f}")
        
        print(f'Epoch [{(i+1)}/{epoch}] OS*: {avg_C_knw:.2f} %')
        print(f'Epoch [{(i+1)}/{epoch}] Unk: {avg_C_unk:.2f} %')
        print(f'Epoch [{(i+1)}/{epoch}] HOS: {hos:.2f} ')
        