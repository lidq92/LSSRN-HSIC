import os
import subprocess
import torch
import rasterio
import numpy as np
from scipy.interpolate import CubicSpline # https://docs.scipy.org/doc/scipy/reference/interpolate.html; https://docs.scipy.org/doc/scipy/tutorial/interpolate.html
from torch.utils.data import Dataset
import Codecs


def process(path, C, Q, D, output_path, BASE_CODEC):
    clipped = rasterio.open(path).read()
    C0, H, W = clipped.shape
    MAXIMUM_VALUE = clipped.max() # clipped.max() + 10 # 10000 # 
    MINIMUM_VALUE = clipped.min() # clipped.min() - 10 # 0 # 
    assert C <= C0 and C > 0 # C = 0 corresponds to INR-based compression
    assert C < C0 or Q < 100 # C = C0 and Q = 100 equals lossless
    indices = np.linspace(0, C0 - 1, C, dtype=int) #
    downsampled_data = clipped[indices, :, :]
    remaining_indices = np.setdiff1d(np.arange(C0), indices)
    remaining_data = clipped[remaining_indices, :, :]
    base_bin_path = output_path.replace(".tif", ".bin")
    recon_path = output_path.replace(".tif", "_recon.tif")
    Codecs.write_tiff_with_rasterio(output_path, downsampled_data)
    NBITS = int(np.ceil(np.log2(downsampled_data.max() + 1)))
    Codecs.encode(output_path, base_bin_path, Q, BASE_CODEC,
                  NBITS, C, H, W)
    subprocess.call(f'rm -f {output_path}', shell=True)
    subprocess.call(f'rm -f {base_bin_path}.aux.xml', shell=True)
    if Q == 100:
        base_data = downsampled_data
    else:
        Codecs.decode(base_bin_path, recon_path, method=BASE_CODEC)
        dataset = rasterio.open(recon_path)
        base_data = dataset.read()
        subprocess.call(f'rm -f {recon_path}', shell=True)
    
    USE_COORDINATES = True # False #
    USE_COLORS = True #
    num_colors = C * (2 * D + 1) ** 2 * USE_COLORS
    num_coords =  2 * USE_COORDINATES
    feature_dim = num_coords + num_colors
    features = np.zeros((H, W, feature_dim), dtype=np.float32) # 
    if USE_COORDINATES:
        coords_h, coords_w = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')    
        ph = 2 * coords_h / (H - 1) - 1
        pw = 2 * coords_w / (W - 1) - 1
        coords = np.stack([ph, pw], axis=-1).astype(np.float32)
        coords = coords.reshape((H, W, -1))
        features[:, :, :num_coords] = coords.reshape(H, W, -1)
    if USE_COLORS:
        base_data_n = (base_data.astype(np.float32) - MINIMUM_VALUE) / (MAXIMUM_VALUE - MINIMUM_VALUE)
        if D > 0:
            base_data_pad = np.pad(base_data_n, 
                            ((0, 0), (D, D), (D, D)),
                            mode='reflect'
                            ).transpose(1, 2, 0) # (H+2D)(W+2D)C
            colors = np.lib.stride_tricks.sliding_window_view(base_data_pad, (2 * D + 1, 2 * D + 1), 
                                                            axis=(0, 1))
            features[:, :, num_coords:] = colors.reshape((H, W, -1))
        else:
            features[:, :, num_coords:] = base_data_n.transpose(1, 2, 0)
    features = features.reshape(H * W, feature_dim)
    if Q == 100:
        out_data = (remaining_data.astype(np.float32) - MINIMUM_VALUE) / (MAXIMUM_VALUE - MINIMUM_VALUE)
        cs = CubicSpline(indices, base_data_n, axis=0, bc_type='natural')
        base_data_interpolated = cs(np.arange(C0))[remaining_indices]
        out_data = out_data - base_data_interpolated
        C1 = C0 - C
    else:
        out_data = (clipped.astype(np.float32) - MINIMUM_VALUE) / (MAXIMUM_VALUE - MINIMUM_VALUE)
        cs = CubicSpline(indices, base_data_n, axis=0, bc_type='natural')
        base_data_interpolated = cs(np.arange(C0))
        out_data = out_data - base_data_interpolated
        C1 = C0
    labels = out_data.transpose(1, 2, 0).reshape(H * W, C1)
    
    return features, labels, MAXIMUM_VALUE, MINIMUM_VALUE


class LSSRNDataset(Dataset):
    def __init__(self, args):
        filename = os.path.splitext(os.path.basename(args.path))[0]
        output_path = f'{args.output_dir}/{filename}_base.tif'
        features, labels, self.MAXIMUM_VALUE, self.MINIMUM_VALUE = process(args.path, args.C, args.Q, args.D, 
                                                                           output_path, args.base_codec)
        self.features = torch.from_numpy(features).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(torch.float32)
        self.n_pixels = len(self.features)
        self.n_feature = self.features.shape[-1]
        self.out_channels = self.labels.shape[-1]
        self.channels = self.out_channels if args.Q < 100 else self.out_channels + args.C
        self.n_subpixels = self.n_pixels * self.channels # H * W * C0

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):  
        feature = self.features[idx]
        label = self.labels[idx]
        
        return feature, label
