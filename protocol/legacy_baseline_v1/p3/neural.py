"""Compact EEGNet-family baseline. Optional PyTorch; exact epoch-boundary resume.

This is a small independently implemented baseline, not a new validated method
or a byte-equivalent implementation of another library's EEGNet.
"""
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import digest, save_arrays, standardized_trials
from .models import balanced_weights

class CompactEEGNet(nn.Module):
    def __init__(self, channels, times, dropout=0.25):
        super().__init__()
        if times < 64: raise ValueError("EEGNet requires at least 64 time points")
        self.features=nn.Sequential(
            nn.Conv2d(1,8,(1,63),padding=(0,31),bias=False), nn.BatchNorm2d(8),
            nn.Conv2d(8,16,(channels,1),groups=8,bias=False), nn.BatchNorm2d(16),
            nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(dropout),
            nn.Conv2d(16,16,(1,15),padding=(0,7),groups=16,bias=False),
            nn.Conv2d(16,16,(1,1),bias=False),nn.BatchNorm2d(16),nn.ELU(),
            nn.AvgPool2d((1,8)),nn.Dropout(dropout))
        self.classifier=nn.Linear(16*(times//32),2)
    def forward(self,x):
        x=self.features(x[:,None,:,:])
        return self.classifier(x.flatten(1))

def atomic_torch(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix="."+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            torch.save(state,f);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def predict(model,X,device,batch_size=256):
    model.eval(); logits=[]
    with torch.no_grad():
        for begin in range(0,len(X),batch_size):
            batch=torch.from_numpy(standardized_trials(X[begin:begin+batch_size])).to(device)
            logits.append(model(batch).cpu().numpy())
    values=np.concatenate(logits)
    exp=np.exp(values-values.max(axis=1,keepdims=True))
    return exp/exp.sum(axis=1,keepdims=True),values

def validation_loss(rows,y,probabilities):
    losses=-np.log(np.clip(probabilities[np.arange(len(y)),y],1e-12,1))
    datasets=sorted({r["dataset"] for r in rows})
    by_dataset=[]
    for dataset in datasets:
        people=sorted({str(r["subject"]) for r in rows if r["dataset"]==dataset})
        by_dataset.append(np.mean([np.mean([losses[i] for i,r in enumerate(rows)
                          if r["dataset"]==dataset and str(r["subject"])==subject]) for subject in people]))
    return float(np.mean(by_dataset))

def train_eegnet(folder,X,y,rows,train_idx,val_idx,config,seed,identity,device="cpu",max_new_epochs=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(int(config.get("threads",4)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model=CompactEEGNet(X.shape[1],X.shape[2],config["dropout"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"],weight_decay=0)
    sampler=torch.Generator(device="cpu");sampler.manual_seed(seed+1000003)
    train_rows=[rows[i] for i in train_idx]
    weights=torch.as_tensor(balanced_weights(train_rows,y[train_idx]),dtype=torch.float64)
    train_X=torch.from_numpy(standardized_trials(X[train_idx]));train_y=torch.from_numpy(y[train_idx])
    state_path=folder/"last.pt"; start=0;history=[];best_loss=float("inf");best_epoch=0;bad_epochs=0
    best_model=None
    if state_path.exists():
        saved=torch.load(state_path,map_location="cpu",weights_only=True)
        if saved["identity_sha256"]!=digest(identity): raise ValueError("Checkpoint/data/code/config identity changed")
        if saved["device"]!=device: raise ValueError("Resume device changed; use a separately recorded new run")
        model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        sampler.set_state(saved["sampler_rng"]);torch.set_rng_state(saved["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start=saved["epoch"];history=saved["history"];best_loss=saved["best_loss"]
        best_epoch=saved["best_epoch"];bad_epochs=saved["bad_epochs"];best_model=saved["best_model"]
    max_epochs=int(config["epochs"])
    has_val=val_idx is not None and len(val_idx)>0
    val_rows=[rows[i] for i in val_idx] if has_val else []
    limit=max_epochs if max_new_epochs is None else min(max_epochs,start+int(max_new_epochs))
    epoch=start
    for epoch in range(start+1,limit+1):
        if has_val and bad_epochs>=config["patience"]: epoch-=1;break
        model.train();total_loss=0.0
        for _ in range(config["steps_per_epoch"]):
            indices=torch.multinomial(weights,config["batch_size"],replacement=True,generator=sampler)
            xb=train_X[indices].to(device);yb=train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(xb),yb)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite neural loss")
            loss.backward();optimizer.step();total_loss+=float(loss.detach().cpu())
        current=None
        if has_val:
            probabilities,logits=predict(model,X[val_idx],device)
            current=validation_loss(val_rows,y[val_idx],probabilities)
            save_arrays(folder/f"validation_epoch_{epoch:04d}.npz",indices=val_idx,y=y[val_idx],
                        probabilities=probabilities,logits=logits)
        if not has_val or current<best_loss-config["min_delta"]:
            best_loss=current if has_val else 0.0;best_epoch=epoch;bad_epochs=0
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad_epochs+=1
        history.append({"epoch":epoch,"updates":epoch*config["steps_per_epoch"],
                        "train_cross_entropy":total_loss/config["steps_per_epoch"],
                        "source_validation_cross_entropy":current})
        state={"identity_sha256":digest(identity),"identity":identity,"device":device,"seed":seed,
               "epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
               "sampler_rng":sampler.get_state(),"torch_rng":torch.get_rng_state(),
               "cuda_rng":torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
               "history":history,"best_loss":best_loss,"best_epoch":best_epoch,
               "bad_epochs":bad_epochs,"best_model":best_model}
        atomic_torch(state_path,state)
        if config.get("retain_epoch_checkpoints",True):
            atomic_torch(folder/f"model_epoch_{epoch:04d}.pt",{"identity_sha256":digest(identity),
                         "epoch":epoch,"model":model.state_dict()})
        print(f"epoch {epoch}/{max_epochs} train_CE={history[-1]['train_cross_entropy']:.5f} "
              f"source_val_CE={current}",flush=True)
    complete=epoch>=max_epochs or (has_val and bad_epochs>=config["patience"])
    if best_model is not None:
        model.load_state_dict(best_model)
        atomic_torch(folder/"selected.pt",{"identity_sha256":digest(identity),"epoch":best_epoch,
                     "model":best_model,"channels":X.shape[1],"n_times":X.shape[2],"config":config})
    return model,{"complete":complete,"epochs_completed":epoch,"selected_epoch":best_epoch,
                  "best_source_validation_loss":best_loss if has_val and best_model is not None else None,
                  "history":history,"checkpoint_resume":"Exact epoch boundary; partial epoch reruns from last.pt"}
