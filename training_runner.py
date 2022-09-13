import argparse
import debugpy
import h5py
import logging
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from WatChMaL.watchmal.engine.engine_classifier import ClassifierEngine


from runner_util import utils, train_config

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--doTraining", help="run training", action="store_true")
parser.add_argument("--doQuickPlots", help="run training", action="store_true")
parser.add_argument("--testParser", help="run training", action="store_true")
parser.add_argument("--plotInput", help="run training")
parser.add_argument("--plotOutput", help="run training")
parser.add_argument("--training_input", help="where training files are")
parser.add_argument("--training_output_path", help="where to dump training output")
args = parser.parse_args(['--training_input','foo','@args_training.txt',
                            '--plotInput','foo','@args_training.txt',
                            '--plotOutput','foo','@args_training.txt',
                            '--training_output_path','foo','@args_training.txt'])

logger = logging.getLogger('train')


def training_runner(rank, settings):

    print(f"rank: {rank}")
    #gpu = settings.gpuNumber[rank]
    world_size=1
    settings.numGPU=1
    if settings.multiGPU:
        world_size = len(settings.gpuNumber)
        settings.numGPU = len(settings.gpuNumber)
        settings.gpuNumber = settings.gpuNumber[rank]

    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    #Initialize training configuration, classifier and dataset for engine training
    settings.set_output_directory()
    settings.initTrainConfig()
    settings.initClassifier()
    settings.initOptimizer()
    #If labels do not start at 0, saves offset so that they are changed during training
    settings.checkLabels()

    data_config, data_loader, train_indices, test_indices, val_indices = settings.initDataset(rank)
    model = nn.parallel.DistributedDataParallel(settings.classification_engine, device_ids=[settings.gpuNumber])
    engine = ClassifierEngine(model, rank, settings.gpuNumber, settings.outputPath)

    engine.configure_data_loaders(data_config, data_loader, settings.multiGPU, 0, train_indices, test_indices, val_indices, settings)
    engine.configure_optimizers(settings)
    settings.save_options(settings.outputPath, 'training_settings')
    engine.train(settings)

def init_training():

    #Choose settings for utils class in util_config.ini
    settings = utils()
    if settings==0:
        print("Settings did not initialize properly, exiting...")
        exit

    os.environ['MASTER_ADDR'] = 'localhost'

    master_port = 12355
        
    # Automatically select port based on base gpu
    os.environ['MASTER_PORT'] = str(master_port)

    if settings.multiGPU:
        master_port += settings.gpuNumber[0]

        mp.spawn(training_runner, nprocs=len(settings.gpuNumber), args=(settings,))
    else:
        training_runner(0, settings)
    
def main():
    if args.doTraining:
        init_training()

    elif args.testParser:
        settings = utils()

    elif args.doQuickPlots:
        fig = disp_learn_hist(args.plotInput, losslim=2, show=False)
        fig.savefig(args.plotOutput+'resnet_test.png', format='png')

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    print("MAIN")
    main()


'''
elif args.doTraining:
    main()




elif args.testParser:
    settings = utils()

elif args.doQuickPlots:
    fig = disp_learn_hist(args.plotInput, losslim=2, show=False)
    fig.savefig(args.plotOutput+'resnet_test.png', format='png')
'''
