#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED=42
BATCH_SIZE=128
EPOCHS=25
LR=1e-3
WD_BASE=1e-4
OUT_DIR=Path("eei_original_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS=[
    "kaiming_wd",
    "eei_global_wd",
    "eei_calibrated_wd",
    "eei_calibrated_wd_x0.5",
    "eei_calibrated_wd_x2.0",
]
DISPLAY={
    "kaiming_wd":"Kaiming + WD 1e-4",
    "eei_global_wd":"EEI + WD 1e-4",
    "eei_calibrated_wd":"EEI + WD calibrado",
    "eei_calibrated_wd_x0.5":"EEI + WD cal x0.5",
    "eei_calibrated_wd_x2.0":"EEI + WD cal x2.0",
}

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(3072,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)
        self.fc4=nn.Linear(256,128)
        self.fc5=nn.Linear(128,64)
        self.fc6=nn.Linear(64,10)
    def forward(self,x):
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        return self.fc6(x)

def layers(model):
    return [(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear)]

def eei_params(model):
    rows=[]
    for i,(name,m) in enumerate(layers(model)):
        fan_in,_=nn.init._calculate_fan_in_and_fan_out(m.weight)
        sigma_k=math.sqrt(2.0/fan_in)
        h_k=0.5*math.log(math.pi*math.e*(sigma_k**2)/2.0)
        rows.append(dict(layer=i+1,layer_name=name,fan_in=fan_in,
                         sigma_kaiming=sigma_k,H_kaiming=h_k))
    h_star=float(np.mean([r["H_kaiming"] for r in rows]))
    sigma_eei=math.sqrt(2.0*math.exp(2.0*h_star)/(math.pi*math.e))
    for r in rows:
        r["H_star"]=h_star
        r["sigma_eei"]=sigma_eei
        r["delta_H"]=h_star-r["H_kaiming"]
        r["wd_calibrated"]=WD_BASE*math.exp(r["delta_H"])
    return sigma_eei,rows

def init_kaiming(model):
    for _,m in layers(model):
        nn.init.kaiming_normal_(m.weight,mode="fan_in",nonlinearity="relu")
        nn.init.zeros_(m.bias)

def init_eei(model):
    sigma,rows=eei_params(model)
    for _,m in layers(model):
        nn.init.normal_(m.weight,0.0,sigma)
        nn.init.zeros_(m.bias)
    return rows

def optimizer_global(model,wd):
    weights=[]; biases=[]
    for _,m in layers(model):
        weights.append(m.weight); biases.append(m.bias)
    return torch.optim.Adam([
        {"params":weights,"weight_decay":wd},
        {"params":biases,"weight_decay":0.0},
    ],lr=LR)

def optimizer_calibrated(model,rows,mult):
    rec={r["layer_name"]:r for r in rows}
    groups=[]
    for name,m in layers(model):
        groups.append({"params":[m.weight],"weight_decay":rec[name]["wd_calibrated"]*mult})
        groups.append({"params":[m.bias],"weight_decay":0.0})
    return torch.optim.Adam(groups,lr=LR)

def make_data():
    t=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    tr=datasets.CIFAR10("./data",train=True,download=True,transform=t)
    te=datasets.CIFAR10("./data",train=False,download=True,transform=t)
    return tr,te

def loaders(tr,te):
    g=torch.Generator(); g.manual_seed(SEED)
    return (
        DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,generator=g,num_workers=2,
                   pin_memory=torch.cuda.is_available()),
        DataLoader(te,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,
                   pin_memory=torch.cuda.is_available()),
    )

def train_epoch(model,loader,opt):
    model.train(); loss_sum=0.; corr=0; n=0
    for x,y in loader:
        x=x.to(DEVICE); y=y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        z=model(x); loss=F.cross_entropy(z,y)
        loss.backward(); opt.step()
        bs=y.size(0); n+=bs
        loss_sum+=loss.item()*bs
        corr+=(z.argmax(1)==y).sum().item()
    return loss_sum/n,corr/n

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); loss_sum=0.; corr=0; n=0
    for x,y in loader:
        x=x.to(DEVICE); y=y.to(DEVICE)
        z=model(x); loss=F.cross_entropy(z,y)
        bs=y.size(0); n+=bs
        loss_sum+=loss.item()*bs
        corr+=(z.argmax(1)==y).sum().item()
    return loss_sum/n,corr/n

def run(method,tr,te):
    seed_all()
    train_loader,test_loader=loaders(tr,te)
    model=MLP().to(DEVICE)
    eei_rows=None
    if method=="kaiming_wd":
        init_kaiming(model); opt=optimizer_global(model,WD_BASE)
    else:
        eei_rows=init_eei(model)
        if method=="eei_global_wd":
            opt=optimizer_global(model,WD_BASE)
        elif method=="eei_calibrated_wd":
            opt=optimizer_calibrated(model,eei_rows,1.0)
        elif method=="eei_calibrated_wd_x0.5":
            opt=optimizer_calibrated(model,eei_rows,0.5)
        elif method=="eei_calibrated_wd_x2.0":
            opt=optimizer_calibrated(model,eei_rows,2.0)
    hist=[]
    for ep in range(1,EPOCHS+1):
        tr_loss,tr_acc=train_epoch(model,train_loader,opt)
        te_loss,te_acc=evaluate(model,test_loader)
        gap=tr_acc-te_acc
        print(f"{DISPLAY[method]:24s} ep {ep:02d}/{EPOCHS} test_acc={te_acc:.4f} gap={gap:.4f}")
        hist.append(dict(method=method,method_name=DISPLAY[method],epoch=ep,
                         train_loss=tr_loss,train_accuracy=tr_acc,
                         test_loss=te_loss,test_accuracy=te_acc,
                         generalization_gap=gap))
    return hist,eei_rows

def main():
    print("RECONSTRUCAO DO BENCHMARK EEI DA DISSERTACAO")
    print("CIFAR-10 | 3072->1024->512->256->128->64->10 | ReLU | Adam 1e-3 | batch 128 | 25 ep | seed 42")
    tr,te=make_data()
    all_hist=[]; param_rows=None
    for method in METHODS:
        h,p=run(method,tr,te); all_hist.extend(h)
        if p is not None: param_rows=p
    df=pd.DataFrame(all_hist)
    df.to_csv(OUT_DIR/"history.csv",index=False)

    summary=[]
    for method,g in df.groupby("method",sort=False):
        g=g.sort_values("epoch")
        peak=g.loc[g["test_accuracy"].idxmax()]
        e1=g.iloc[0]
        summary.append(dict(
            method=DISPLAY[method],
            acc_epoch1_percent=100*e1["test_accuracy"],
            peak_accuracy_percent=100*peak["test_accuracy"],
            peak_epoch=int(peak["epoch"]),
            gap_peak_percent=100*peak["generalization_gap"],
        ))
    sdf=pd.DataFrame(summary)
    sdf.to_csv(OUT_DIR/"summary.csv",index=False)

    if param_rows is not None:
        pd.DataFrame(param_rows).to_csv(OUT_DIR/"eei_parameters.csv",index=False)

    plt.figure(figsize=(9,5))
    for method,g in df.groupby("method",sort=False):
        plt.plot(g["epoch"],100*g["test_accuracy"],label=DISPLAY[method])
    plt.xlabel("Epoca"); plt.ylabel("Acuracia de teste (%)")
    plt.title("EEI - Acuracia de teste"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR/"figure14_accuracy.png",dpi=180); plt.close()

    plt.figure(figsize=(9,5))
    for method,g in df.groupby("method",sort=False):
        plt.plot(g["epoch"],100*g["generalization_gap"],label=DISPLAY[method])
    plt.xlabel("Epoca"); plt.ylabel("Gap treino - teste (p.p.)")
    plt.title("EEI - Gap de generalizacao"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR/"figure14_gap.png",dpi=180); plt.close()

    print("\nResumo:")
    print(sdf.to_string(index=False))

if __name__=="__main__":
    main()
