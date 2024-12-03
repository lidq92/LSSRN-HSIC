import os
import sys
import time
import struct
import random
import warnings
import argparse
import subprocess
import tracemalloc
import fpzip
import torch
import rasterio
import numpy as np
# from scipy.ndimage import zoom
from scipy.interpolate import CubicSpline
import Codecs
import logger
from LSSRNmodel import LSSRNModel


warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CODECS = ['JPEG2000', 'JPEGXL'] #


def read_image_header(bitstream):
    ptr = 0
    n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    BASE_CODEC = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    C = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    D = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    C0 = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    Q = struct.unpack('<f',bitstream[ptr: ptr + 4])[0]
    ptr += 4
    bcnl = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    bc = 2 ** (bcnl >> 4)
    nl = bcnl & 0x0F
    MAXIMUM_VALUE = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=True)
    ptr += 2
    MINIMUM_VALUE = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=True)
    ptr += 2
    nn_bytes = int.from_bytes(bitstream[ptr: ptr + 3], byteorder='big', signed=False)

    return n_bytes_header, BASE_CODEC, C, D, C0, Q, bc, nl, MAXIMUM_VALUE, MINIMUM_VALUE, nn_bytes
    
        
def test(bitstream, dirname, filename, BASE_CODEC):
    sub_nn_bitstream = bitstream[:nn_bytes]
    sub_nn_bitstream_path = f'{dirname}/{filename}_nn.bin'
    with open(sub_nn_bitstream_path, 'wb') as f_out: f_out.write(sub_nn_bitstream)
    bitstream = bitstream[nn_bytes:]
    
    base_recon_path = f'{dirname}/{filename}_base_recon.tif'
    base_bin_path = f'{dirname}/{filename}_base.bin'
    with open(base_bin_path, 'wb') as f_out: f_out.write(bitstream)
    BASE_CODEC = BASE_CODECS[BASE_CODEC]
    Codecs.decode(base_bin_path, base_recon_path, method=BASE_CODEC)

    dataset = rasterio.open(base_recon_path)
    base = dataset.read()  # CHW or HW
    subprocess.call(f'rm -f {base_recon_path}', shell=True)
    base = base.reshape((-1, base.shape[-2], base.shape[-1])) # CHW
    C, H, W = base.shape
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
        # base_n = base.astype(np.float32) / base.max()
        base_n = (base.astype(np.float32) - MINIMUM_VALUE) / (MAXIMUM_VALUE - MINIMUM_VALUE)
        if D > 0:
            base_pad = np.pad(base_n, 
                            ((0, 0), (D, D), (D, D)),
                            mode='reflect'
                            ).transpose(1, 2, 0) # (H+2D)(W+2D)C
            colors = np.lib.stride_tricks.sliding_window_view(base_pad, (2 * D + 1, 2 * D + 1), axis=(0, 1))
            features[:, :, num_coords:] = colors.reshape((H, W, -1))
        else:
            features[:, :, num_coords:] = base_n.transpose(1, 2, 0)
    features = features.reshape(H * W, feature_dim)
    if Q == 100:
        C1 = C0 - C
    else:
        C1 = C0

    model = LSSRNModel(
        dim_in=features.shape[-1], 
        dim_hidden=bc,
        dim_out=C1,
        num_layers=nl,
        # activation=torch.nn.ReLU(), # Default: Sine
        final_activation=torch.nn.Tanh() # torch.nn.Identity() # Default: Sigmoid
    )
    model = model.to(DEVICE)
    
    with open(sub_nn_bitstream_path,'rb') as f: compressed_bytes = f.read()
    params = fpzip.decompress(compressed_bytes, order='C')[0][0][0]
    k = 0
    state_dict = {}
    for param_tensor in model.state_dict():
        values = params[k:k+model.state_dict()[param_tensor].numel()].reshape(model.state_dict()[param_tensor].size())
        state_dict[param_tensor] = torch.from_numpy(values)
        k = k + model.state_dict()[param_tensor].numel()
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(features).to(torch.float32)
        y_pred = torch.zeros(x.shape[0], C1).to(DEVICE)
        y_pred = model(x.to(DEVICE))
        if Q == 100:
            image = y_pred.to('cpu').numpy() # 
            image = image.reshape(H, W, C1)
            image = np.transpose(image, axes=(2, 0, 1))
            recon = np.zeros((C0, H, W), dtype=np.uint16)
            indices = np.linspace(0, C0 - 1, C, dtype=int)
            cs = CubicSpline(indices, base_n, axis=0, bc_type='natural')
            remaining_indices = np.setdiff1d(np.arange(C0), indices)
            base_interpolated = cs(np.arange(C0))[remaining_indices]
            image = base_interpolated + image
            image = np.round((image) * (MAXIMUM_VALUE - MINIMUM_VALUE) + MINIMUM_VALUE).astype(np.uint16)
            recon[indices] = base
            recon[remaining_indices] = image
            recon = np.clip(recon, a_min=0, a_max=10000) #
            Codecs.write_tiff_with_rasterio(recon_path, recon)
        else:
            indices = np.linspace(0, C0 - 1, C, dtype=int) #
            cs = CubicSpline(indices, base_n, axis=0, bc_type='natural')
            base_interpolated = cs(np.arange(C0))
            image = y_pred.to('cpu').numpy() # 
            image = image.reshape(H, W, C1)
            image = np.transpose(image, axes=(2, 0, 1))
            image = image + base_interpolated
            image = np.round(image * (MAXIMUM_VALUE - MINIMUM_VALUE) + MINIMUM_VALUE)
            image = np.clip(image, a_min=0, a_max=10000) #
            image = image.astype(np.uint16)
            Codecs.write_tiff_with_rasterio(recon_path, image)
        logger.log.info(f'Recon: {recon_path}')
        subprocess.call(f'rm -f {base_bin_path}', shell=True)
        subprocess.call(f'rm -f {base_bin_path}.aux.xml', shell=True)
        subprocess.call(f'rm -f {sub_nn_bitstream_path}', shell=True)
    
    return bitstream


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSSRN-HSIC')
    parser.add_argument('--seed', type=int, default=19920517)
    parser.add_argument('-i', '--bin_path', type=str, help='binstream path')
    parser.add_argument('-org', '--org_path', type=str, default=None, help='org path')
    args = parser.parse_args()
    torch.manual_seed(args.seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    # tracemalloc.start()

    dirname, basename = os.path.split(args.bin_path)
    filename = os.path.splitext(basename)[0]
    if os.path.exists(f'{dirname}/decode.txt'):
        decoded = False
        with open(f'{dirname}/decode.txt', 'r') as file:
            content = file.read()
            if "bpsp" in content:
                decoded = True
                print('Bitstream already decoded!')
        if decoded:
            sys.exit()
    logger.create_logger(dirname, 'decode.txt')
    logger.log.info(f'Binstream: {args.bin_path}')
    start_time = time.time()
    with open(args.bin_path, 'rb') as fin: bitstream = fin.read()

    n_bytes_header, BASE_CODEC, C, D, C0, Q, bc, nl, MAXIMUM_VALUE, MINIMUM_VALUE, nn_bytes = read_image_header(bitstream)
    bitstream = bitstream[n_bytes_header:]
    
    bin_path = args.bin_path
    recon_path = f'{dirname}/{basename[:-4]}_recon.tif'

    bitstream = test(bitstream, dirname, filename, BASE_CODEC)

    end_time = time.time()
    logger.log.info(f'Time elapsed: {end_time - start_time}')

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # logger.log.info(f"Current memory usage: {current / 10**6:.2f} MB")
    # logger.log.info(f"Peak memory usage: {peak / 10**6:.2f} MB")

    if args.org_path is not None:
        org_img = rasterio.open(args.org_path).read()
        rec_img = rasterio.open(recon_path).read()
        bytes = os.path.getsize(bin_path)
        mse_value = np.mean((org_img.astype(np.float32) - rec_img.astype(np.float32)) ** 2) #
        logger.log.info(f"MSE: {mse_value}")
        peak = np.max(org_img.astype(np.float32)) # 
        psnr = 10 * np.log10(peak ** 2 / mse_value)
        logger.log.info(f"PSNR: {psnr}")   
        n_subpixels = np.prod(org_img.shape)
        logger.log.info(f"Total size: {bytes} bytes, bpsp={bytes * 8 / n_subpixels}")
        if True: # False: # Delete the reconstructed image?
            subprocess.call(f'rm -f {recon_path}', shell=True)
    