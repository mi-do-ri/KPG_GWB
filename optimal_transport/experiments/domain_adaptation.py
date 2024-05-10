from ._experiment import Experiment
from ..models._ot import _OT

from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.svm import SVC

class DomainAdaptation(Experiment):
    def __init__(
        self,
        model: Dict[str, _OT],
        exp_name: str,
        log_dir: str,
        # classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name, log_dir)

    def class_balancing(self, x, t, num=10, seed=0):
        t = np.squeeze(t)
        x_class = []
        n_class = t.max()
        for k in range(1, n_class + 1, 1):
            x_class.append(x[t == k])
        np.random.seed(seed)
        x_class_balance = [xx[np.random.choice(np.arange(len(xx)), num, replace=True)] for xx in x_class]
        t_class_balance = [np.ones(num, dtype=np.int32) * k for k in range(1, n_class + 1, 1)]
        x_class_balance = np.vstack(x_class_balance)
        t_class_balance = np.hstack(t_class_balance)
        return x_class_balance, t_class_balance

    def load_data(self, source, target, num_labeled=1, class_balance=True, num=10, seed=0):
        source_data = sio.loadmat("F:/DS Lab/OT/KPG_GWB/notebook/data/hda/decaf/{}_fc6.mat".format(source))
        target_data = sio.loadmat("F:/DS Lab/OT/KPG_GWB/notebook/data/hda/resnet50/{}.mat".format(target))

        source_feat, source_label = source_data["fts"], source_data["labels"]
        target_feat, target_label = target_data["fts"], target_data["labels"]
        source_label, target_label = source_label.reshape(-1, ), target_label.reshape(-1, )

        indexes = sio.loadmat("F:/DS Lab/OT/KPG_GWB/notebook/data/hda/labeled_index/{}_{}.mat".format(target,num_labeled))
        idx_labeled,idx_unlabeled = indexes["labeled_index"][0], indexes["unlabeled_index"][0]

        target_feat_l, target_label_l = target_feat[idx_labeled], target_label[idx_labeled]
        target_feat_un, target_label_un = target_feat[idx_unlabeled], target_label[idx_unlabeled]

        if class_balance:
            source_feat, source_label = self.class_balancing(source_feat, source_label, num=num,seed=seed)
            target_feat_un, target_label_un = self.class_balancing(target_feat_un, target_label_un, num=num)

        return source_feat, source_label, target_feat_l, target_label_l, target_feat_un, target_label_un

    def run(
        self, feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, model: _OT, **kwargs
    ) -> np.ndarray:
        I = []
        J = []
        t = 0
        feat_sl = []

        
        for l in label_tl:
            I.append(t)
            J.append(t)
            fl = feat_s[label_s==l]
            feat_sl.append(np.mean(fl, axis=0))
            t += 1
            
        feat_sl = np.vstack(feat_sl)
        label_s_ = np.concatenate((label_tl, label_s))
        feat_s_ = np.vstack((feat_sl, feat_s))
        
        feat_t_ = np.vstack((feat_tl, feat_tu))
        
        p = np.ones(len(feat_s_))/len(feat_s_)
        q = np.ones(len(feat_t_))/len(feat_t_)
        K = list(zip(I, J))

        model.fit(feat_s_, label_s_, feat_t_, p, q, K, **kwargs)
        return model.transport(feat_s_, feat_t_)
                    

    def plot(
        self, x_axis: str, y_axis: str,
        save_fig: bool = True, **kwargs
    ):
        plt.figure(figsize=(12, 8))
        for algo, record in self.record_[self.exp_name].items():
            plt.plot(record[x_axis], record[y_axis], label=algo)
        
        plt.title(self.exp_name)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend()
        plt.grid(True)

        if save_fig:
            plt.savefig(os.path.join(self.log_dir, f"{self.cur_time}.png"), dpi=300)
        plt.show()

    
    @classmethod
    def accuracy(
        cls, feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu
    ) -> float:
        feat_train = np.vstack((feat_tl,feat_s_trans[len(feat_tl):]))
        label_train = np.hstack((label_tl,label_s))
        # print("train svm...")
        clf = SVC(gamma='auto',probability=True)
        clf.fit(feat_train,label_train)
        acc = clf.score(feat_tu,label_tu)
        return acc
    
    @classmethod
    def keypoints(
        cls, X: np.ndarray, y: np.ndarray, 
        keypoints_per_cls: int = 1
    ) -> List:
        def euclidean(source, target, p=2):
            return np.sum(
                np.power(
                    source.reshape([source.shape[0], 1, source.shape[1]]) -
                    target.reshape([1, target.shape[0], target.shape[1]]),
                    p
                ),
                axis=-1
            ) ** 1/2
        
        labels = np.unique(y)
        selected_inds = []
        for label in labels:
            cls_indices = np.where(y == label)[0]
            distance = euclidean(X[cls_indices], np.mean(X[cls_indices], axis=0)[None, :]).squeeze()
            selected_inds.extend(cls_indices[np.argsort(distance)[:keypoints_per_cls]])
        return selected_inds
    

class HDA(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        **kwargs
    ):
        super().__init__(model, exp_name="HDA", log_dir=log_dir, **kwargs)
    
    def __call__(self):
        self.record_[self.exp_name] = {model_id: {"samples": [], "accuracy": [], "runtime": []} for model_id in self.model}

        domains = ["amazon", "dslr", "webcam"]
        num_labeled = 1
        seed = 1
        i = 0
        
        for source in domains:
            for target in domains:
                
                feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = self.load_data(source, target, num_labeled, seed=seed)

                self.logger.info(f"Source:{source}, Target: {target}")
                
                for model_id in self.model:
                    start = time.time()
                    feat_s_trans = self.run(feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, self.model[model_id])
                    
                    accuracy = DomainAdaptation.accuracy(feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu)
                    self.record_[self.exp_name][model_id]["accuracy"].append(accuracy)
                    self.record_[self.exp_name][model_id]["runtime"].append(time.time() - start)
                    self.logger.info(f"[{model_id}] Accuracy: {accuracy}, Runtime: {time.time() - start}")

                self.checkpoint()

        # Calculate mean accuracy for each model_id after all iterations
        for model_id in self.model:
            mean_accuracy = sum(self.record_[self.exp_name][model_id]["accuracy"]) / len(self.record_[self.exp_name][model_id]["accuracy"])
            self.logger.info(f"Mean accuracy for {model_id}: {mean_accuracy}")
        
        # self.plot(x_axis="samples", y_axis="accuracy")
        return self.record_[self.exp_name]
