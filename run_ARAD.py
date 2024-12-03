import os
import csv
import warnings
import resource
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
from rasterio.errors import NotGeoreferencedWarning

import Codecs
import encode
import decode


MAX_WORKERS = 16 # int( 2 * os.cpu_count() // 3) #
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft_limit = 65535 #
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def run_one_rate_point(base_codec, file_path, C, i, start_id=901):   
    
    subprocess.call(f'python encode.py -i {file_path} -C {C} -codec {base_codec}', shell=True)
    bin_path = f'outputs/ARAD_1K_{i+start_id:04d}_{base_codec}_C{C}_Q100.0_D0_bc32_nl1_prec32_lr0.001_bs32_e10/ARAD_1K_{i+start_id:04d}.bin'
    subprocess.call(f'python decode.py -i {bin_path}', shell=True)
    recon_path = f'outputs/ARAD_1K_{i+start_id:04d}_{base_codec}_C{C}_Q100.0_D0_bc32_nl1_prec32_lr0.001_bs32_e10/ARAD_1K_{i+start_id:04d}_recon.tif'

    mse_value, psnr, bits, bpsp = Codecs.eval_RD(bin_path, recon_path, file_path)
    
    subprocess.call(f'rm -f {bin_path} {recon_path}', shell=True)
    print(f"file: {file_path} @C={C}, MSE: {mse_value}, PSNR: {psnr}, bits: {bits}, bpsp: {bpsp}")
    
    return file_path, C, psnr, bpsp, mse_value, bits, i


def main(start_id=901, end_id_plus_1=951):
    Cs = range(5, 31, 3)
    base_codec = 'JPEGXL'
    
    file_paths = [
        f"data/ARAD/ARAD_1K_{idx:04d}.mat" for idx in range(start_id, end_id_plus_1)
    ]
    output_paths = [
        f"data/ARAD/ARAD_1K_{idx:04d}.tif" for idx in range(start_id, end_id_plus_1)
    ]

    NBITS = 12
    for i, file_path in enumerate(file_paths):
        if os.path.exists(output_paths[i]):
            continue
        file = h5py.File(file_path, 'r')
        hsi = file['cube'][:].transpose(0, 2, 1)
        hsi = hsi.clip(min=0, max=1) # Handling ARAD_1K_0929
        org_img = np.round(hsi * (2 ** NBITS - 1)).astype('<u2')
        Codecs.write_tiff_with_rasterio(output_paths[i], org_img) # preprocessing to tif

    SOTA_results_dir = 'SOTA_results'
    os.makedirs(SOTA_results_dir, exist_ok=True)

    csv_file = f'{SOTA_results_dir}/test_LSSRN-HSIC.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        metrics = ['PSNR', 'bpsp', 'MSE', 'bits']
        
        csv_headers = ['Path'] + [f"C={C}_{metric}" for C in Cs for metric in metrics]
        writer.writerow(csv_headers)

        futures = []
        global MAX_WORKERS
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for i, file_path in enumerate(output_paths): 
                for c_idx, C in enumerate(Cs):
                    futures.append(executor.submit(run_one_rate_point, base_codec, file_path, C, i, start_id))
                    
        results = {}
        for future in as_completed(futures):
            try:
                file_path, C, psnr, bpsp, mse_value, bits, i = future.result()
                if file_path not in results:
                    results[file_path] = [None] * (len(Cs) * len(metrics))
                c_idx = Cs.index(C)
                results[file_path][len(metrics) * c_idx] = psnr
                results[file_path][len(metrics) * c_idx + 1] = bpsp
                results[file_path][len(metrics) * c_idx + 2] = mse_value
                results[file_path][len(metrics) * c_idx + 3] = bits
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        for file_path in output_paths:
            row = [file_path] + results.get(file_path, [None] * (len(Cs) * len(metrics)))
            writer.writerow(row)


if __name__ == "__main__":
    main()
