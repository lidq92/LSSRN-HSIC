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
import numpy as np
from ignite.engine import Events
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from rasterio.errors import NotGeoreferencedWarning

import logger
from LSSRNloss import LSSRNLoss
from LSSRNmodel import LSSRNModel
from LSSRNdataset import LSSRNDataset
from LSSRNperformance import LSSRNPerformance
from modified_ignite_engine import create_supervised_evaluator, create_supervised_trainer


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CODECS = ['JPEG2000', 'JPEGXL'] #


def write_image_header(header_path, 
                       BASE_CODEC, C, D, C0, Q, bc, nl, 
                       MAXIMUM_VALUE, MINIMUM_VALUE, nn_bytes):
    n_bytes_header  = 0
    n_bytes_header += 1      # Number of bytes header
    n_bytes_header += 1      # BASE_CODECS.index
    n_bytes_header += 1      # C
    n_bytes_header += 1      # D
    n_bytes_header += 1      # C0
    n_bytes_header += 4      # Q
    n_bytes_header += 1      # log2(bc) (4bits), nl (4bits) 
    n_bytes_header += 2      # MAXIMUM_VALUE
    n_bytes_header += 2      # MINIMUM_VALUE
    n_bytes_header += 3      # Number of bytes nn
    byte_to_write   = b''
    byte_to_write  += n_bytes_header.to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += BASE_CODECS.index(BASE_CODEC).to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += C.to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += D.to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += C0.to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += struct.pack('<f', Q)
    byte_to_write  += (int(np.log2(bc)) * 2 ** 4 + nl).to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += int(MAXIMUM_VALUE).to_bytes(2, byteorder='big', signed=True)
    byte_to_write  += int(MINIMUM_VALUE).to_bytes(2, byteorder='big', signed=True)
    byte_to_write += nn_bytes.to_bytes(3, byteorder='big', signed=False)
    with open(header_path, 'wb') as fout: fout.write(byte_to_write)
    if n_bytes_header != os.path.getsize(header_path):
        raise ValueError(f'Invalid number of bytes in header! '
                         f'expected {n_bytes_header}, got {os.path.getsize(header_path)}')


def train(args):
    dataset = LSSRNDataset(args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=32, pin_memory=True)  
    model = LSSRNModel(
        dim_in=dataset.n_feature, 
        dim_hidden=args.base_channel, 
        dim_out=dataset.out_channels,
        num_layers=args.num_layers,
        # activation=torch.nn.ReLU(), # Default: Sine
        final_activation=torch.nn.Tanh() # torch.nn.Identity() # Default: Sigmoid
    )
    model = model.to(DEVICE)
    logger.log.info(model)
    for param_tensor in model.state_dict():
        logger.log.info('{}\t {}'.format(param_tensor, model.state_dict()[param_tensor].size()))
    total_params = sum(p.numel() for p in model.parameters())
    logger.log.info('total_params: {}'.format(total_params))

    optimizer = Adam(model.parameters(), lr=args.lr) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, int(args.epochs/2)), gamma=0.1) #
    loss_func = LSSRNLoss()
    trainer = create_supervised_trainer(model, optimizer, loss_func, device=DEVICE)
    evaluator = create_supervised_evaluator(model, metrics={'LSSRN_performance': LSSRNPerformance()}, device=DEVICE)
    writer = SummaryWriter(log_dir=args.output_dir)
    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = 1e8, -1 # MSE
    filename = os.path.splitext(os.path.basename(args.path))[0]
    @trainer.on(Events.ITERATION_COMPLETED)
    def iter_event_function(engine):
        writer.add_scalar(f'train/loss/{filename}', engine.state.output, engine.state.iteration)
    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_event_function(engine):
        scheduler.step()
        global best_val_criterion, best_epoch
        if args.epochs == 1:
            torch.save(model.state_dict(), f'{args.output_dir}/model.pt')
            best_epoch = engine.state.epoch
            return
        if engine.state.epoch % min(args.val_duration, args.epochs) == 0: 
            evaluator.run(train_loader)
            performance = evaluator.state.metrics
            writer.add_scalar(f'val/MSE/{filename}', performance['MSE'], engine.state.epoch)
            val_criterion = performance['MSE'] * (dataset.MAXIMUM_VALUE - dataset.MINIMUM_VALUE).astype(np.float32) ** 2 #
            if val_criterion < best_val_criterion: 
                torch.save(model.state_dict(), f'{args.output_dir}/model.pt')
                best_val_criterion = val_criterion
                best_epoch = engine.state.epoch
                logger.log.info('Save current best val model (MSE: {:.5f}) @epoch {}'
                                .format(best_val_criterion, best_epoch))
            else:
                logger.log.info('Model is not updated (MSE: {:.5f}) @epoch: {}'
                                .format(val_criterion, engine.state.epoch))           
    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        logger.log.info('best epoch: {}'.format(best_epoch))
        model.load_state_dict(torch.load(f'{args.output_dir}/model.pt', weights_only=True))
        subprocess.call(f'rm -f {args.output_dir}/model.pt', shell=True)
        params = None
        for param_tensor in model.state_dict(): #
            if params is None:
                params = model.state_dict()[param_tensor].data.to('cpu').numpy().reshape(-1)
            else:
                params = np.concatenate((params, model.state_dict()[param_tensor].data.to('cpu').numpy().reshape(-1)))
        compressed_bytes = fpzip.compress(params, precision=args.precision, order='C')
        nn_path = f'{args.output_dir}/{filename}_nn.bin'
        with open(nn_path, 'wb') as f: f.write(compressed_bytes)
        nn_bytes = os.path.getsize(nn_path)
        nn_bpsp = nn_bytes * 8 / dataset.n_subpixels
        logger.log.info(f'nn: {nn_bytes} bytes, bpsp={nn_bpsp}')
        base_bin_path = f'{args.output_dir}/{filename}_base.bin'
        base_bytes = os.path.getsize(base_bin_path)
        base_bpsp = base_bytes * 8 / dataset.n_subpixels
        logger.log.info(f"base: {base_bytes} bytes: bpsp={base_bpsp}")
        writer.close()  

    trainer.run(train_loader, max_epochs=args.epochs)

    return dataset.MAXIMUM_VALUE, dataset.MINIMUM_VALUE, dataset.channels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSSRN-HSIC')
    parser.add_argument('--seed', type=int, default=19920517)
    parser.add_argument('-rn', '--randomness', action='store_true',
                        help='Allow randomness during training?')
    parser.add_argument('-i', '--path', type=str,
                        help='path of input tif or img file')
    parser.add_argument('-o', '--output_dir', default='outputs', type=str,
                        help='output dir')
    parser.add_argument('-codec', '--base_codec', default='JPEGXL', type=str,
                        help='Base codec (Default: JPEGXL)')
    parser.add_argument('-C', '--C', type=int, default=30,
                        help=' (default: 30)')
    parser.add_argument('-Q', '--Q', type=float, default=100.0,
                        help=' (default: 100.0)')
    parser.add_argument('-D', '--D', type=int, default=0,
                        help='#neighbors (2D+1)^2, default D: 0')
    parser.add_argument('-bc', '--base_channel', type=int, default=32,
                        help='base channel (default: 32)')
    parser.add_argument('-nl', '--num_layers', type=int, default=1,
                        help='Number of layers (default: 1)')
    parser.add_argument('-prec', '--precision', type=int, default=32,
                        help=' (default: 32)')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-vd', '--val_duration', type=int, default=1,
                        help='number of epoch duration for val (default: 1)')
    args = parser.parse_args()

    # tracemalloc.start()

    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    org_path = args.path
    filename = os.path.splitext(os.path.basename(org_path))[0]
    fs = '{}/{}_{}_C{}_Q{}_D{}_bc{}_nl{}_prec{}_lr{}_bs{}_e{}'
    args.output_dir = fs.format(args.output_dir, filename, args.base_codec, args.C, args.Q, args.D, 
                                args.base_channel, args.num_layers, args.precision,
                                args.lr, args.batch_size, args.epochs)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    bitstream_path = f'{args.output_dir}/{filename}.bin'
    if os.path.exists(f'{args.output_dir}/encode.txt'):
        encoded = False
        with open(f'{args.output_dir}/encode.txt', 'r') as file:
            content = file.read()
            if "Time elapsed" in content:
                encoded = True
                print('Bitstream already created!')
        if encoded and os.path.exists(bitstream_path):
        # if encoded:
            sys.exit()
    logger.create_logger(args.output_dir, 'encode.txt')
    start_time = time.time()
    header_path = f'{bitstream_path}_header'

    logger.log.info(args)
    MAXIMUM_VALUE, MINIMUM_VALUE, C0 = train(args)
    nn_bitstream_path = f'{args.output_dir}/{filename}_nn.bin'
    nn_bytes = os.path.getsize(nn_bitstream_path)
    base_bitstream_path = f'{args.output_dir}/{filename}_base.bin'
    subprocess.call(f'rm -f {header_path}', shell=True)
    write_image_header(header_path, args.base_codec, args.C, args.D, C0, args.Q, 
                       args.base_channel, args.num_layers, 
                       MAXIMUM_VALUE, MINIMUM_VALUE, nn_bytes
                       )
    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {header_path}', shell=True)
    subprocess.call(f'cat {nn_bitstream_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {nn_bitstream_path}', shell=True)
    subprocess.call(f'cat {base_bitstream_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {base_bitstream_path}', shell=True)

    end_time = time.time()
    logger.log.info(f'Time elapsed: {end_time - start_time}')

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # logger.log.info(f"Current memory usage: {current / 10**6:.2f} MB")
    # logger.log.info(f"Peak memory usage: {peak / 10**6:.2f} MB")
