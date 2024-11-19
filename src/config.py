import argparse
from pathlib import Path
import pprint

username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}
def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei'],
                        help='dataset to use (default: mosi)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)')

    parser.add_argument('--dropout_tse', type=float, default=0.1,
                        help='dropout of tse')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    parser.add_argument('--d_prjh', type=int, default=32,
                        help='hidden size in projection network,32 for mosi 128 for mosei')
    parser.add_argument('--dropout_r', type=float, default=0.4)
    parser.add_argument('--multi_head', type=int, default=8)

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 32 for mosi 128 for mosei)')
    parser.add_argument('--loss_num', type=int, default=4,
                        help='number of loss')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='warm step ratio (default: 0.1)')
    parser.add_argument('--lr_main', type=float, default=1e-3,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_bert', type=float, default=5e-5,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--lr_et', type=float, default=2e-3,
                        help='initial learning rate for DBIT parameters (default: 3e-5)')
    parser.add_argument('--lr_te', type=float, default=2e-3,
                        help='initial learning rate for TSE parameters (default: 3e-5)')

    parser.add_argument('--weight_decay_main', type=float, default=1e-3,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the bert Adam optimizer')
    parser.add_argument('--weight_decay_et', type=float, default=1e-3,
                        help='L2 penalty factor of the DBIT Adam optimizer')
    parser.add_argument('--weight_decay_te', type=float, default=1e-3,
                        help='L2 penalty factor of the TSE Adam optimizer')

    parser.add_argument('--optim', type=str, default='AdamW',
                        help='optimizer to use (default: AdamW)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')

    # Logistics
    parser.add_argument('--tse_layers', type=int, default=2,
                        help='TSE layers')
    parser.add_argument('--ETlayers', type=int, default=3,
                        help='Encoder Tower layers')
    parser.add_argument('--step_ratio', type=int, default=5,
                        help='frequency of result logging (default: 50 for mosi 150 for mosei)')
    parser.add_argument('--a_size', type=int, default=50,
                        help='frequency of result logging (default: 50 for mosi 150 for mosei)')
    parser.add_argument('--v_size', type=int, default=50,
                        help='frequency of result logging (default: 50 for mosi 150 for mosei)')
    parser.add_argument('--dislen', type=int, default=30,
                        help='dislen (default: 30 for mosi 60 for mosei)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    args = parser.parse_args()
    return args


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        """self.__dict__包括了self的内含属性"""
        return config_str


def get_config(dataset, mode, batch_size):
    config = Config(data=dataset, mode=mode)

    config.dataset = dataset
    config.batch_size = batch_size

    return config