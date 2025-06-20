# LSSRN-HSIC
    
## Requirements
- Python 3.10.12
- rasterio
- fpzip
- GDAL 3.10.0 # JPEG XL Driver etc. https://jpegxl.info/resources/battle-of-codecs.htm
```bash
git clone https://github.com/OSGeo/gdal.git # '3.11.0dev-a8117d90bf'
cd gdal && mkdir build && cd build
# sudo apt install swig
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -DNDEBUG" -DBUILD_PYTHON_BINDINGS=ON ..
make -j
sudo make install
# ...
```

## Preparation: Data and Base Codecs 
### Data
```bash
# download ARAD 1K dataset from https://github.com/boazarad/ARAD_1K
ln -s ARAD_1K_HSI_abs_path data/ARAD
```
data/ARAD/ARAD_1K_09xx.tif

### Base Codecs
See Codecs.py.

## Encode
```bash
python encode.py -i data/ARAD/ARAD_1K_0901.tif
```

## Decode
```bash
python decode.py -i outputs/ARAD_1K_0901_JPEGXL_C30_Q100.0_D0_bc32_nl1_prec32_lr0.001_bs32_e10/ARAD_1K_0901.bin -org data/ARAD/ARAD_1K_0901.tif
```

## Experiments
```bash
python Codecs.py # run SOTA codecs (including preprocessing of the datasets)
python run_ARAD.py # run LoCaCF-HSIC: SOTA + LoCaCF
# Code of LineRWKV: https://github.com/diegovalsesia/linerwkv
# Code of MST++: https://github.com/caiyuanhao1998/MST-plus-plus/
# We provide the results of MST++ with JPEG XL, LineRWKV (L, XS) in SOTA_results
python BD_metrics.py # BD metrics and R-D curves
```

## TODO
- AVIF (compression of 8/10/12-bits grayscale images), etc. https://gdal.org/en/latest/drivers/raster/avif.html
- HEIF with GDAL, https://gdal.org/en/latest/drivers/raster/heif.html

## FAQ

## Contact
Dingquan Li @ Pengcheng Laboratory, dingquanli AT pku DOT edu DOT cn.
