import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import random
from scipy.stats import nbinom
from skimage import io, color, transform
import warnings
from PIL import Image

# internal helping functions

def _rpoispp(lambda_val, win):
    
    # 计算区域的面积
    area = (win[1] - win[0]) * (win[3] - win[2])
    
    # 计算预期的点数
    expected_points = np.random.poisson(lambda_val * area)
    
    # 生成点的坐标
    x_coords = np.random.uniform(win[0], win[1], expected_points)
    y_coords = np.random.uniform(win[2], win[3], expected_points)
    
    return np.column_stack((x_coords, y_coords))

def _simu_zi(family, subject_n, zi_p=0.5, mu=0.5, size=0.25):
    Y = np.empty(subject_n)
    ind_mix = np.random.binomial(1, zi_p, size=subject_n)
    
    if family == "ZIP":
        Y[ind_mix != 0] = 0
        Y[ind_mix == 0] = np.random.poisson(mu, size=np.sum(ind_mix == 0))
    elif family == "ZINB":
        Y[ind_mix != 0] = 0
        Y[ind_mix == 0] = nbinom.rvs(n=size, p=1 / (1 + mu / size), size=np.sum(ind_mix == 0))
    
    return Y


def simulate_gene(lambda_val=0.7, spots=10000, se=50, ns=50,start=0, type='ZINB', se_p=0.3, 
            se_size=10, se_mu=10, ns_p=0.3, ns_size=5, ns_mu=5, ptn='2_ring.png',label=[0,1],
            png_dir='/home/gongyuqiao/ur_annotation/Mytrain/amplification/ptn_png', outlier=False):
    
    """
    Simulate gene expression data based on a Poisson point process using image's pattern.

    Parameters:
    - lambda_val: Average number of points per unit area.
    - spots: Total number of points to simulate.
    - se: The number of  spatially variable genes (SVGs).
    - ns: The number of  non-SVGs.
    - type: Type of distribution for expression simulation ('ZINB' or 'ZIP').
    - se_p: Probability of zero-inflated expression for SVGs in the streak area.
    - se_size: Size parameter for zero-inflated distribution for SVGs in the streak area.
    - se_mu: For SVGs, the lambda para in the poisson distribution or the mu para in the NB distribution.
    - ns_p: Probability of zero-inflated expression of non-SVGs and SVGs in the non-streak area.
    - ns_size: For non-SVGs and SVGs in the non-streak area, the size para in the NB distribution.
    - ns_mu: For non-SVGs and SVGs in the non-streak area, the lambda para in the poisson
             distribution or the mu para in the NB distribution.
    - ptn: The file name of the pattern png image.
    - png_dir: Directory where the image is located.
    - outlier: Whether to simulate outliers in the expression data.

    Returns:
    - adata: AnnData object containing the simulated gene expression data and spatial coordinates.
    """

    win_size = int(np.ceil(np.sqrt(spots / lambda_val)))
    win = [0, win_size, 0, win_size]
    coor_x = _rpoispp(lambda_val, win)
    coor_dt = pd.DataFrame({
    'row': coor_x[:,0].astype(int),
    'col': coor_x[:,1].astype(int)
    })
    coor_dt = coor_dt.drop_duplicates().reset_index(drop=True)
    coor_dt['cell'] = ['c_'+str(i) for i in range(coor_dt.shape[0])]

    # Load and process the image
    image_path = f"{png_dir}/{ptn}"
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    re_img = transform.resize(gray_image, (win_size, win_size))

    # Convert the image into a binary mask
    img_coor = np.round(re_img)
    img_coords = np.argwhere(img_coor > 0)

    # Merge coordinates
    coor_s1 = coor_dt.merge(pd.DataFrame(img_coords, columns=['row', 'col']), on=['row', 'col'])

    # Extract marked and random coordinates
    coor_mark = coor_s1
    coor_random = coor_dt[~coor_dt.cell.isin(coor_mark.cell)]

    # Simulate expression for marked coordinates
    exp_mark = np.array([_simu_zi(family=type, subject_n=len(coor_mark), zi_p=se_p, size=se_size, mu=se_mu) for _ in range(se)]).T
    
    # Simulate expression for random coordinates
    exp_random = np.array([_simu_zi(family=type, subject_n=len(coor_random), zi_p=ns_p, size=ns_size, mu=ns_mu) for _ in range(se)]).T
    
    # Combine expression data
    exp_svg = np.vstack((exp_mark, exp_random))
    non_coor = pd.concat([coor_mark, coor_random])
    
    # Simulate non-SVG expression data
    exp_non = np.array([_simu_zi(family=type, subject_n=len(non_coor), zi_p=ns_p, size=ns_size, mu=ns_mu) for _ in range(ns)]).T

    # Combine all data
    all_data = np.hstack((non_coor[['row', 'col']], exp_non[:,:start],exp_svg, exp_non[:,start:]))

    # Handle outliers
    if outlier:
        if outlier < 0 or outlier >= 1:
            print("# outlier parameter is wrong!")
            end = all_data
        else:
            ind = random.sample(range(len(all_data)), round(len(all_data) * outlier))
            out_para = 5
            for idx in ind:
                all_data[idx, 2:] = _simu_zi(family=type, subject_n=(len(ind) * (all_data.shape[1] - 2)), 
                                              zi_p=se_p / 2, size=se_size * out_para, mu=se_mu * out_para)
            end = all_data
    else:
        end = all_data
    
    # Create a DataFrame for the results
    result_df = pd.DataFrame(end)
    adata = sc.AnnData(X=result_df.iloc[:,2:])
    adata.obsm['spatial'] = result_df.iloc[:,:2].to_numpy()
    adata.obs['mark_area'] = [label[0]]*coor_mark.shape[0] + [label[1]]*coor_random.shape[0]
    return adata
    return adata