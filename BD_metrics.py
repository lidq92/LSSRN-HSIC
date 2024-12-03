# https://github.com/Anserw/Bjontegaard_metric
import csv
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)
    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)
    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)
    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))
    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)
    # find avg diff
    avg_diff = (int2 - int1) / (max_int - min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)
    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)
    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))
    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)
    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100

    return avg_diff


def read_csv(csv_file_path, N=50, start_k=1, end_k_p1=7):
    rps = end_k_p1 - start_k
    psnrs, bpsp_values, bits = np.zeros((N, rps)), np.zeros((N, rps)), np.zeros((N, rps))
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  
        r = 0
        for row in reader:
            for k in range(start_k, end_k_p1):
                psnrs[r, k - start_k] = float(row[4 * k + 1])
                bpsp_values[r, k - start_k] = float(row[4 * k + 2]) 
                # mse_values[r, k - start_k] = float(row[4 * k + 3]) 
                bits[r, k - start_k] = float(row[4 * k + 4])  
            r = r + 1
            if r == N: break #

        return psnrs, bpsp_values, bits


def SOTA():
    N = 50 # 
    csv_file_path = f'SOTA_results/test_LSSRN-HSIC.csv'
    LSSRN_psnrs, LSSRN_bpsp_values, LSSRN_bits_values = read_csv(csv_file_path, N, start_k=0, end_k_p1=9)

    csv_file_path = f'SOTA_results/test_JPEG2000.csv'
    jpeg2000_psnrs, jpeg2000_bpsp_values, jpeg2000_bits_values = read_csv(csv_file_path, N, start_k=1, end_k_p1=10)

    csv_file_path = f'SOTA_results/test_JPEGXL.csv'
    jpegxl_psnrs, jpegxl_bpsp_values, jpegxl_bits_values = read_csv(csv_file_path, N, start_k=1, end_k_p1=10)

    csv_file_path = f'SOTA_results/test_MST++_JPEGXL_Hybrid.csv'
    MSTpp_psnrs, MSTpp_bpsp_values, MSTpp_bits_values = read_csv(csv_file_path, N, start_k=0, end_k_p1=9)

    csv_file_path = f'SOTA_results/test_LineRWKV_xs.csv'
    LineRWKV_xs_psnrs, LineRWKV_xs_bpsp_values, LineRWKV_xs_bits_values = read_csv(csv_file_path, N, start_k=1, end_k_p1=9)

    csv_file_path = f'SOTA_results/test_LineRWKV_l.csv'
    LineRWKV_l_psnrs, LineRWKV_l_bpsp_values, LineRWKV_l_bits_values = read_csv(csv_file_path, N, start_k=1, end_k_p1=9)
    
    bd_rate = BD_RATE(jpeg2000_bits_values.mean(axis=0), jpeg2000_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    bd_psnr = BD_PSNR(jpeg2000_bits_values.mean(axis=0), jpeg2000_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    print(f'vs. JPEG 2000: AVG BD-PSNR={bd_psnr}, AVG BD-Rate={bd_rate}')

    bd_rate = BD_RATE(jpegxl_bits_values.mean(axis=0), jpegxl_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    bd_psnr = BD_PSNR(jpegxl_bits_values.mean(axis=0), jpegxl_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    print(f'vs. JPEG XL: AVG BD-PSNR={bd_psnr}, AVG BD-Rate={bd_rate}')

    bd_rate = BD_RATE(MSTpp_bits_values.mean(axis=0), MSTpp_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    bd_psnr = BD_PSNR(MSTpp_bits_values.mean(axis=0), MSTpp_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    print(f'vs. MST++ with JPEG XL: AVG BD-PSNR={bd_psnr}, AVG BD-Rate={bd_rate}')

    bd_rate = BD_RATE(LineRWKV_xs_bits_values.mean(axis=0), LineRWKV_xs_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    bd_psnr = BD_PSNR(LineRWKV_xs_bits_values.mean(axis=0), LineRWKV_xs_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    print(f'vs. LineRWKV_xs: AVG BD-PSNR={bd_psnr}, AVG BD-Rate={bd_rate}')

    bd_rate = BD_RATE(LineRWKV_l_bits_values.mean(axis=0), LineRWKV_l_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    bd_psnr = BD_PSNR(LineRWKV_l_bits_values.mean(axis=0), LineRWKV_l_psnrs.mean(axis=0),
                      LSSRN_bits_values.mean(axis=0), LSSRN_psnrs.mean(axis=0))
    print(f'vs. LineRWKV_l: AVG BD-PSNR={bd_psnr}, AVG BD-Rate={bd_rate}')

    plt.figure(figsize=(8, 6)) 
    plt.plot(jpeg2000_bpsp_values.mean(axis=0), jpeg2000_psnrs.mean(axis=0), 
                label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpegxl_bpsp_values.mean(axis=0), jpegxl_psnrs.mean(axis=0), 
                label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
    plt.plot(MSTpp_bpsp_values.mean(axis=0), MSTpp_psnrs.mean(axis=0), 
                label='MST++ with JPEG XL', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LineRWKV_xs_bpsp_values.mean(axis=0), LineRWKV_xs_psnrs.mean(axis=0), 
                label='LineRWKV (XS)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LineRWKV_l_bpsp_values.mean(axis=0), LineRWKV_l_psnrs.mean(axis=0), 
                label='LineRWKV (L)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LSSRN_bpsp_values.mean(axis=0), LSSRN_psnrs.mean(axis=0), 
                label='LSSRN-HSIC', linestyle='-', marker='o', alpha=0.9)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(np.linspace(0, 5, 11), fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.tight_layout()
    plt.savefig(f'SOTA_results/Average_RD_Curve_SOTA.pdf', dpi=300) 
    # plt.savefig(f'SOTA_results/Average_RD_Curve_SOTA.png', dpi=300) 
    plt.close()

if __name__ == "__main__":
    SOTA()
