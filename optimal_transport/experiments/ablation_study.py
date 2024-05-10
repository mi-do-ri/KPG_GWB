from .domain_adaptation import DomainAdaptation
from ..models._ot import _OT
from ..models import KeypointGW

from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt


class AlphaSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        **kwargs
    ):
        super().__init__(model, exp_name="mix_sensitive", log_dir=log_dir, **kwargs)

    def __call__(
        self,
        min_alpha: float = 0.0,
        max_alpha: float = 1.0,
        freq_alpha: float = 0.1,
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"guide_mix": [], "accuracy": [], "mean_accuracy": [], "runtime": []} for model_id in self.model}

        guide_mixes = np.arange(min_alpha, max_alpha+freq_alpha, freq_alpha)
        for guide_mix in guide_mixes:
            start = time.time()
            self.model["KeypointGW"].alpha = guide_mix
                
            domains = ["amazon", "dslr", "webcam"]
            num_labeled = 1
            seed = 1

            for source in domains:
                for target in domains:
                    
                    feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = self.load_data(source, target, num_labeled, seed=seed)

                    # self.logger.info(f"Source:{source}, Target: {target}")

                    feat_s_trans = self.run(feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, self.model["KeypointGW"])
                    
                    accuracy = DomainAdaptation.accuracy(feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu)
                    self.record_[self.exp_name]["KeypointGW"]["accuracy"].append(accuracy)

            mean_accuracy = sum(self.record_[self.exp_name]["KeypointGW"]["accuracy"]) / len(self.record_[self.exp_name]["KeypointGW"]["accuracy"])

            self.record_[self.exp_name]["KeypointGW"]["guide_mix"] \
                .append(guide_mix)
            self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"] \
                .append(mean_accuracy)
            self.record_[self.exp_name]["KeypointGW"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{int(guide_mix*100)}% guiding] Accuracy: {self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointGW"]["runtime"][-1]}')

        self.plot(x_axis="guide_mix", y_axis="mean_accuracy")
        return self.record_[self.exp_name]


class BarycenterSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        **kwargs
    ):
        super().__init__(model, exp_name="barycenter_sensitive", log_dir=log_dir, **kwargs)

    def  __call__(
        self,
        min_barycenter,
        max_barycenter,
        num_classes,
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"multiples": [], "accuracy": [], "mean_accuracy": [], "runtime": []} for model_id in self.model}
        
        n_barycenters = np.arange(min_barycenter, max_barycenter+num_classes, num_classes)
        for k in n_barycenters:
            start = time.time()
            self.model["KeypointGW"].num_free_barycenters = k
                
            domains = ["amazon", "dslr", "webcam"]
            num_labeled = 1
            seed = 1

            for source in domains:
                for target in domains:
                    
                    feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = self.load_data(source, target, num_labeled, seed=seed)

                    # self.logger.info(f"Source:{source}, Target: {target}")

                    feat_s_trans = self.run(feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, self.model["KeypointGW"])
                    
                    accuracy = DomainAdaptation.accuracy(feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu)
                    self.record_[self.exp_name]["KeypointGW"]["accuracy"].append(accuracy)

            mean_accuracy = sum(self.record_[self.exp_name]["KeypointGW"]["accuracy"]) / len(self.record_[self.exp_name]["KeypointGW"]["accuracy"])
            multiples = int(k // num_classes)
            self.record_[self.exp_name]["KeypointGW"]["multiples"] \
                .append(multiples)
            self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"] \
                .append(mean_accuracy)
            self.record_[self.exp_name]["KeypointGW"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{multiples} times] Accuracy: {self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointGW"]["runtime"][-1]}')

        self.plot(x_axis="multiples", y_axis="mean_accuracy")
        return self.record_[self.exp_name]
    

class EpsilonSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        **kwargs
    ):
        super().__init__(model, exp_name="epsilon_sensitive", log_dir=log_dir, **kwargs)
    
    def __call__(
        self,
        eps_range: List = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    ):
        self.record_[self.exp_name] = {model_id: {"eps": [], "accuracy": [], "mean_accuracy": [], "runtime": []} for model_id in self.model}
        
        for eps in eps_range:
            start = time.time()
            self.model["KeypointGW"].sinkhorn_reg = eps
                
            domains = ["amazon", "dslr", "webcam"]
            num_labeled = 1
            seed = 1

            for source in domains:
                for target in domains:
                    
                    feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = self.load_data(source, target, num_labeled, seed=seed)

                    # self.logger.info(f"Source:{source}, Target: {target}")

                    feat_s_trans = self.run(feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, self.model["KeypointGW"])
                    
                    accuracy = DomainAdaptation.accuracy(feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu)
                    self.record_[self.exp_name]["KeypointGW"]["accuracy"].append(accuracy)

            mean_accuracy = sum(self.record_[self.exp_name]["KeypointGW"]["accuracy"]) / len(self.record_[self.exp_name]["KeypointGW"]["accuracy"])

            self.record_[self.exp_name]["KeypointGW"]["eps"] \
                .append(eps)
            self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"] \
                .append(mean_accuracy)
            self.record_[self.exp_name]["KeypointGW"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{eps} entropy] Accuracy: {self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointGW"]["runtime"][-1]}')

        self.plot(x_axis="eps", y_axis="mean_accuracy")
        return self.record_[self.exp_name]  

class RhoSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        **kwargs
    ):
        super().__init__(model, exp_name="rho_sensitive", log_dir=log_dir, **kwargs)

    def __call__(
        self,
        rho_range: List = [0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5],
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"temperature": [], "accuracy": [], "mean_accuracy": [], "runtime": []} for model_id in self.model}

        for rho in rho_range:
            start = time.time()
            self.model["KeypointGW"].temperature = rho
                
            domains = ["amazon", "dslr", "webcam"]
            num_labeled = 1
            seed = 1

            for source in domains:
                for target in domains:
                    
                    feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = self.load_data(source, target, num_labeled, seed=seed)

                    # self.logger.info(f"Source:{source}, Target: {target}")

                    feat_s_trans = self.run(feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, self.model["KeypointGW"])
                    
                    accuracy = DomainAdaptation.accuracy(feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu)
                    self.record_[self.exp_name]["KeypointGW"]["accuracy"].append(accuracy)

            mean_accuracy = sum(self.record_[self.exp_name]["KeypointGW"]["accuracy"]) / len(self.record_[self.exp_name]["KeypointGW"]["accuracy"])

            self.record_[self.exp_name]["KeypointGW"]["temperature"] \
                .append(rho)
            self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"] \
                .append(mean_accuracy)
            self.record_[self.exp_name]["KeypointGW"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{rho} temperature] Accuracy: {self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointGW"]["runtime"][-1]}')

        self.plot(x_axis="temperature", y_axis="mean_accuracy")
        return self.record_[self.exp_name]  
    
class TolSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        **kwargs
    ):
        super().__init__(model, exp_name="tol_sensitive", log_dir=log_dir, **kwargs)

    def __call__(
        self,
        tol_range: List = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1.5e-5, 1e-4, 1e-3, 1e-2],
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"stop_thr": [], "accuracy": [], "mean_accuracy": [], "runtime": []} for model_id in self.model}

        for tol in tol_range:
            start = time.time()
            self.model["KeypointGW"].tol = tol
                
            domains = ["amazon", "dslr", "webcam"]
            num_labeled = 1
            seed = 1

            for source in domains:
                for target in domains:
                    
                    feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = self.load_data(source, target, num_labeled, seed=seed)

                    # self.logger.info(f"Source:{source}, Target: {target}")

                    feat_s_trans = self.run(feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, self.model["KeypointGW"])
                    
                    accuracy = DomainAdaptation.accuracy(feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu)
                    self.record_[self.exp_name]["KeypointGW"]["accuracy"].append(accuracy)

            mean_accuracy = sum(self.record_[self.exp_name]["KeypointGW"]["accuracy"]) / len(self.record_[self.exp_name]["KeypointGW"]["accuracy"])

            self.record_[self.exp_name]["KeypointGW"]["stop_thr"] \
                .append(tol)
            self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"] \
                .append(mean_accuracy)
            self.record_[self.exp_name]["KeypointGW"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{tol} stop_thr] Accuracy: {self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointGW"]["runtime"][-1]}')

        self.plot(x_axis="stop_thr", y_axis="mean_accuracy")
        return self.record_[self.exp_name]  
    
class ItersSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        **kwargs
    ):
        super().__init__(model, exp_name="sinkhorn_iter_sensitive", log_dir=log_dir, **kwargs)

    def __call__(
        self,
        iters_range: List = [5, 10, 50, 100, 200, 500, 1000, 2000],
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"iters": [], "accuracy": [], "mean_accuracy": [], "runtime": []} for model_id in self.model}

        for iters in iters_range:
            start = time.time()
            self.model["KeypointGW"].sinkhorn_max_iters = iters
                
            domains = ["amazon", "dslr", "webcam"]
            num_labeled = 1
            seed = 1

            for source in domains:
                for target in domains:
                    
                    feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = self.load_data(source, target, num_labeled, seed=seed)

                    # self.logger.info(f"Source:{source}, Target: {target}")

                    feat_s_trans = self.run(feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, self.model["KeypointGW"])
                    
                    accuracy = DomainAdaptation.accuracy(feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu)
                    self.record_[self.exp_name]["KeypointGW"]["accuracy"].append(accuracy)

            mean_accuracy = sum(self.record_[self.exp_name]["KeypointGW"]["accuracy"]) / len(self.record_[self.exp_name]["KeypointGW"]["accuracy"])

            self.record_[self.exp_name]["KeypointGW"]["iters"] \
                .append(iters)
            self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"] \
                .append(mean_accuracy)
            self.record_[self.exp_name]["KeypointGW"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{iters} iters] Accuracy: {self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointGW"]["runtime"][-1]}')

        self.plot(x_axis="iters", y_axis="mean_accuracy")
        return self.record_[self.exp_name]  
    
class LearningRateSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        **kwargs
    ):
        super().__init__(model, exp_name="lr_sensitive", log_dir=log_dir, **kwargs)

    def __call__(
        self,
        lr_range: List = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1],
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"learning_rate": [], "accuracy": [], "mean_accuracy": [], "runtime": []} for model_id in self.model}

        for lr in lr_range:
            start = time.time()
            self.model["KeypointGW"].learning_rate = lr
                
            domains = ["amazon", "dslr", "webcam"]
            num_labeled = 1
            seed = 1

            for source in domains:
                for target in domains:
                    
                    feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu = self.load_data(source, target, num_labeled, seed=seed)

                    # self.logger.info(f"Source:{source}, Target: {target}")

                    feat_s_trans = self.run(feat_s, label_s, feat_tl, label_tl, feat_tu, label_tu, self.model["KeypointGW"])
                    
                    accuracy = DomainAdaptation.accuracy(feat_s, feat_s_trans, label_s, feat_tl, label_tl, feat_tu, label_tu)
                    self.record_[self.exp_name]["KeypointGW"]["accuracy"].append(accuracy)

            mean_accuracy = sum(self.record_[self.exp_name]["KeypointGW"]["accuracy"]) / len(self.record_[self.exp_name]["KeypointGW"]["accuracy"])

            self.record_[self.exp_name]["KeypointGW"]["learning_rate"] \
                .append(lr)
            self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"] \
                .append(mean_accuracy)
            self.record_[self.exp_name]["KeypointGW"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{lr} learning_rate] Accuracy: {self.record_[self.exp_name]["KeypointGW"]["mean_accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointGW"]["runtime"][-1]}')

        self.plot(x_axis="learning_rate", y_axis="mean_accuracy")
        return self.record_[self.exp_name]  