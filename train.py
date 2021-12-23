import matplotlib.pyplot as plt
import time
import sys
import os
import torch
from solver_test import Solver

from ptflops import get_model_complexity_info
from data import AudioDataLoader, AudioDataset
from models.model_50_SFR import FaSNet_base
import args_parameter as parser
plt.switch_backend('agg')


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    sys.path.append("..")
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    # print("cuda is available=", torch.cuda.is_available())
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
          "cuda name is {} ".format(torch.cuda.get_device_name()))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
          "current_device is {} ".format(torch.cuda.current_device()))

    # Construct Solver
    # data
    tr_dataset = AudioDataset(
                              args.train_dir,
                              args.batch_size,
                              sample_rate=args.sample_rate,
                              segment=args.segment
                              )
    cv_dataset = AudioDataset(
                              args.valid_dir,
                              batch_size=4,                    # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segment=args.segment,            # -1 -> use full audio
                              cv_maxlen=args.cv_maxlen
                              )
    tr_loader = AudioDataLoader(
                                tr_dataset,
                                batch_size=1,
                                shuffle=args.shuffle
                                )
    cv_loader = AudioDataLoader(cv_dataset,
                                batch_size=1
                                )
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    if args.model_type in ['dprnn']:
        # model = FaSNet_base(**dprnn_conf)
        model = FaSNet_base(enc_dim=args.enc_dim,
                            feature_dim=args.feature_dim,
                            hidden_dim=args.hidden_dim,
                            layer=args.layer,
                            segment_size=args.segment_size,
                            nspk=args.nspk,
                            win_len=args.win_len)
        # print("Model parameter : {}".format(dprnn_conf))
    elif args.model_type in ['tstnn']:
        model = Net(L=args.segment ,
                    width=args.feature_dim,
                    num_layers=args.layer,)
    else:
        print("please input right model type")
        raise RuntimeError("Unsupported model type : {}".format(args.model))

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), model)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
          "This is model parameters:{} Mb".format(str(check_parameters(model))))
    if args.use_cuda:
        # DataParallel set
        if torch.cuda.device_count() > 1:
            model.to(device)
            model = torch.nn.DataParallel(model)
        else:
            model.cuda()
        # macs, params = get_model_complexity_info(model, (1, 32000), as_strings=True,
        #                                          print_per_layer_stat=True, verbose=True)
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        #       '{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        #       '{:<30}  {:<8}'.format('Number of parameters: ', params))
    torch.set_num_threads(6)
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay
                                     )

    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)

    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


def check_parameters(net):
    """Returns module parameters. Mb"""
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


if __name__ == '__main__':
    # args = parser.parse_args()
    args = parser.get_args()
    print(args)
    main(args)
