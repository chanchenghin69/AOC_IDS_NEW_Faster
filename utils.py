import torch
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import scipy.optimize as opt
import torch.distributions as dist
from sklearn.metrics import accuracy_score
from sklearn.mixture import BayesianGaussianMixture

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        super(SplitData, self).__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            # Preparing the labels
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)
            # abnormal data is labeled as 1, normal data 0
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')

        elif self.dataset == 'unsw':
            # UNSW dataset processing
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        elif self.dataset == 'cic':
            y_ = X[labels].values.astype('float32')
            X_ = X.drop(labels, axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        # Normalization
        normalize = MinMaxScaler().fit(X_)
        x_ = normalize.transform(X_)

        return x_, y_

def description(data):
    print("Number of samples(examples) ",data.shape[0]," Number of features",data.shape[1])
    print("Dimension of data set ",data.shape)

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()

        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class CRCLoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(CRCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):        
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples
        # create mask 
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size).to(self.device)  
        logits_without_ii = logits * logits_mask
        
        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:,(labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:,(labels > 0).squeeze()]
        
        ## This is the denominator for InfoNCE loss: ONE time of traversal
        # sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), axis=1, keepdims=True)
        ## This is the denominator for our proposed CRC loss: TWO times of traversal
        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal))
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)
  
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
    
def score_detail(y_test,y_test_pred,if_print=False):
    # Confusion matrix
    if if_print == True:
        print("Confusion matrix")
        print(confusion_matrix(y_test, y_test_pred))
        # Accuracy 
        print('Accuracy ',accuracy_score(y_test, y_test_pred))
        # Precision 
        print('Precision ',precision_score(y_test, y_test_pred))
        # Recall
        print('Recall ',recall_score(y_test, y_test_pred))
        # F1 score
        print('F1 score ',f1_score(y_test,y_test_pred))

    return accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test,y_test_pred)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    return -np.sum(np.log(0.5 * pdf1 + 0.5 * pdf2))

def fit_bgmm(X, n_components=10, reg_covar=1e-4, random_state=5009):
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        reg_covar=reg_covar,
        max_iter=300,
        init_params='kmeans',
        random_state=random_state
    )
    bgmm.fit(X)
    return bgmm

def capped_sample(X, max_samples, seed=5009):
    if len(X) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx]

@torch.no_grad()
def evaluate(
    normal_temp,
    normal_recon_temp,
    x_train,
    y_train,
    x_test,
    y_test,
    model,
    get_confidence=False,
    en_or_de=False,
    return_predictions=False,
    n_components=10,
    reg_covar=1e-4,
    random_state=5009,
    max_fit_samples=5000
):
    # 兼容 tensor / numpy
    if isinstance(y_train, torch.Tensor):
        y_train_np = y_train.detach().cpu().numpy()
    else:
        y_train_np = np.asarray(y_train)

    normal_mask = (y_train_np == 0).squeeze()
    abnormal_mask = (y_train_np == 1).squeeze()

    # 前向：train
    train_enc, train_dec = model(x_train)
    train_features = F.normalize(train_enc, p=2, dim=1).detach().cpu().numpy()
    train_recon = F.normalize(train_dec, p=2, dim=1).detach().cpu().numpy()

    train_features_normal = train_features[normal_mask]
    train_features_abnormal = train_features[abnormal_mask]
    train_recon_normal = train_recon[normal_mask]
    train_recon_abnormal = train_recon[abnormal_mask]
    
    train_features_normal = capped_sample(train_features_normal, max_fit_samples, random_state)
    train_features_abnormal = capped_sample(train_features_abnormal, max_fit_samples, random_state + 1)
    train_recon_normal = capped_sample(train_recon_normal, max_fit_samples, random_state + 2)
    train_recon_abnormal = capped_sample(train_recon_abnormal, max_fit_samples, random_state + 3)

    # 前向：test
    test_enc, test_dec = model(x_test)
    test_features = F.normalize(test_enc, p=2, dim=1).detach().cpu().numpy()
    test_recon = F.normalize(test_dec, p=2, dim=1).detach().cpu().numpy()

    # ========== Encoder branch: BGMM ==========
    bgmm_feat_normal = fit_bgmm(
        train_features_normal,
        n_components=n_components,
        reg_covar=reg_covar,
        random_state=random_state
    )
    bgmm_feat_abnormal = fit_bgmm(
        train_features_abnormal,
        n_components=n_components,
        reg_covar=reg_covar,
        random_state=random_state
    )

    logp_feat_normal = bgmm_feat_normal.score_samples(test_features)
    logp_feat_abnormal = bgmm_feat_abnormal.score_samples(test_features)

    y_test_pred_en = (logp_feat_abnormal > logp_feat_normal).astype("int32")
    y_test_pro_en = np.abs(logp_feat_abnormal - logp_feat_normal).astype("float32")

    # ========== Decoder branch: BGMM ==========
    bgmm_recon_normal = fit_bgmm(
        train_recon_normal,
        n_components=n_components,
        reg_covar=reg_covar,
        random_state=random_state
    )
    bgmm_recon_abnormal = fit_bgmm(
        train_recon_abnormal,
        n_components=n_components,
        reg_covar=reg_covar,
        random_state=random_state
    )

    logp_recon_normal = bgmm_recon_normal.score_samples(test_recon)
    logp_recon_abnormal = bgmm_recon_abnormal.score_samples(test_recon)

    y_test_pred_de = (logp_recon_abnormal > logp_recon_normal).astype("int32")
    y_test_pro_de = np.abs(logp_recon_abnormal - logp_recon_normal).astype("float32")

    # ========== 融合 ==========
    y_test_pred_final = np.where(
        y_test_pro_en > y_test_pro_de,
        y_test_pred_en,
        y_test_pred_de
    ).astype("int32")

    # 只预测，不算指标
    if isinstance(y_test, int):
        return torch.from_numpy(y_test_pred_final)

    # 评估
    if isinstance(y_test, torch.Tensor):
        y_test_np = y_test.detach().cpu().numpy()
    else:
        y_test_np = np.asarray(y_test)

    result_encoder = score_detail(y_test_np, y_test_pred_en)
    result_decoder = score_detail(y_test_np, y_test_pred_de)
    result_final = score_detail(y_test_np, y_test_pred_final, if_print=True)

    if return_predictions:
        return result_encoder, result_decoder, result_final, y_test_pred_final
    return result_encoder, result_decoder, result_final
