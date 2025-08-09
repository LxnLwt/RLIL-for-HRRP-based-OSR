import argparse
import os
import time
import torch
import numpy as np
import torch.optim as optim
import random
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from model import AlexNet, resnet18
from FSBN import FSBN
from utils.dataset import my_dataloader
from utils.train_eval_utils import train_one_epoch, evaluate, evaluate_auroc_dist


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def main(args):
    para_lambda_list = args.para_lambda_lil
    for para_lambda in para_lambda_list:
        weight_decay_list = args.weight_decay
        for weight_decay in weight_decay_list:
            data_number_list = args.data_number
            for data_number in data_number_list:
                loss_f_list = args.loss_f
                for loss_f in loss_f_list:
                    backbone_list = args.backbone
                    for backbone in backbone_list:
                        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

                        print(args)
                        '''训练日志'''

                        tb_writer = SummaryWriter(log_dir=args.logs_path)
                        nw = 0
                        batch_size = args.batch_size

                        if data_number == 1:
                            train_path = args.train_path1
                            val_path = args.val_path1
                            unknown_path = args.unknown_path1
                            known_path = args.known_path1
                            class_indices_json_path = args.class_indices_json_path1
                            roc_indices_json_path = args.roc_indices_json_path1
                            num_classes = args.num_classes1

                        if data_number == 2:
                            train_path = args.train_path2
                            val_path = args.val_path2
                            unknown_path = args.unknown_path2
                            known_path = args.known_path2
                            class_indices_json_path = args.class_indices_json_path2
                            roc_indices_json_path = args.roc_indices_json_path2
                            num_classes = args.num_classes2

                        if data_number == 3:
                            train_path = args.train_path3
                            val_path = args.val_path3
                            unknown_path = args.unknown_path3
                            known_path = args.known_path3
                            class_indices_json_path = args.class_indices_json_path3
                            roc_indices_json_path = args.roc_indices_json_path3
                            num_classes = args.num_classes3

                        if data_number == 4:
                            train_path = args.train_path4
                            val_path = args.val_path4
                            unknown_path = args.unknown_path4
                            known_path = args.known_path4
                            class_indices_json_path = args.class_indices_json_path4
                            roc_indices_json_path = args.roc_indices_json_path4
                            num_classes = args.num_classes4

                        print('Using {} dataloader workers every process'.format(nw))
                        '''实例化训练集验证集,并加载'''
                        train_dataloader, train_size = my_dataloader(train_path, class_indices_json_path, batch_size=batch_size, nw=nw)
                        val_dataloader, val_size = my_dataloader(val_path, class_indices_json_path, batch_size=args.val_size, nw=nw)
                        known_dataloader, known_size = my_dataloader(known_path, roc_indices_json_path, batch_size=args.roc_size, nw=nw)
                        known_dataloader_oscr, _ = my_dataloader(known_path, class_indices_json_path, batch_size=args.roc_size, nw=nw)
                        unknown_dataloader, _ = my_dataloader(unknown_path, roc_indices_json_path, batch_size=args.roc_size, nw=nw)

                        print("using {} HRRP datas for training, {} HRRP data for validation.".format(train_size, val_size))
                        if backbone == 'alexnet':
                            model = AlexNet(loss_f=loss_f, num_classes=num_classes).to(device)  # 修改模型
                        if backbone == 'resnet34':
                            model = resnet34(loss_f=loss_f, num_classes=num_classes).to(device)  # 修改模型
                        if backbone == 'fsbn':
                            layer_parameter_list = \
                                [
                                    [
                                        [(1, 3, 2), (1, 5, 3), (1, 5, 5), (1, 7, 7), (1, 9, 11), (1, 9, 13), (1, 7, 17), (1, 5, 19),
                                         (1, 5, 23),
                                         (1, 5, 29), (1, 5, 31), (1, 5, 37), (1, 5, 41), (1, 5, 43), (1, 5, 47), (1, 5, 53), (1, 5, 59),
                                         (1, 5, 61), (1, 15, 67), (1, 15, 71), (1, 15, 73), (1, 15, 79), (1, 25, 83), (1, 25, 89)],
                                        [(210, 3, 2), (210, 5, 3), (210, 5, 5), (210, 7, 7), (210, 9, 11), (210, 9, 13),
                                         (210, 7, 17), (210, 5, 19), (210, 5, 23), (210, 5, 29), (210, 5, 31), (210, 5, 37),
                                         (210, 5, 41), (210, 5, 43), (210, 5, 47), (210, 5, 53), (210, 5, 59), (210, 5, 61),
                                         (210, 15, 67), (210, 15, 71), (210, 15, 73), (210, 15, 79), (210, 25, 83), (210, 25, 89)],
                                        [(210, 128, 1), (210, 128, 2)]
                                    ],
                                    [
                                        [(1, 7, 2), (1, 7, 3), (1, 7, 5), (1, 7, 7), (1, 7, 11), (1, 7, 13), (1, 7, 17),
                                         (1, 17, 19), (1, 17, 23), (1, 17, 29), (1, 17, 31), (1, 17, 37), (1, 17, 41),
                                         (1, 17, 43), (1, 12, 47), (1, 12, 53), (1, 11, 59), (1, 11, 61)],
                                        [(214, 7, 2), (214, 7, 3), (214, 7, 5), (214, 7, 7), (214, 7, 11), (214, 7, 13), (214, 7, 17),
                                         (214, 17, 19), (214, 17, 23), (214, 17, 29), (214, 17, 31), (214, 17, 37), (214, 17, 41),
                                         (214, 17, 43), (214, 12, 47), (214, 12, 53), (214, 11, 59), (214, 11, 61)],
                                        [(214, 128, 1), (214, 128, 2)]
                                    ],
                                    [
                                        [(1, 7, 2), (1, 19, 3), (1, 19, 5), (1, 19, 7), (1, 19, 11), (1, 19, 13), (1, 9, 17),
                                         (1, 9, 19),
                                         (1, 17, 23), (1, 25, 29), (1, 25, 31)],
                                        [(187, 7, 2), (187, 19, 3), (187, 19, 5), (187, 19, 7), (187, 19, 11), (187, 19, 13),
                                         (187, 19, 17),
                                         (187, 19, 19), (187, 19, 23), (187, 25, 29), (187, 25, 31)],
                                        [(209, 128, 1), (209, 128, 2)]
                                    ],
                                    [
                                        [(1, 7, 2), (1, 17, 3), (1, 37, 5), (1, 27, 7), (1, 27, 11), (1, 17, 13), (1, 17, 17)],
                                        [(149, 7, 2), (149, 17, 3), (149, 37, 5), (149, 27, 7), (149, 27, 11), (149, 17, 13),
                                         (149, 17, 17)],
                                        [(149, 128, 1), (149, 128, 2)]
                                    ]
                                ]
                            model = FSBN(layer_parameter_list=layer_parameter_list, loss_f=loss_f, n_class=num_classes).to(device)  # 修改模型
                        if args.weights != "":
                            if os.path.exists(args.weights):
                                weights_dict = torch.load(args.weights, map_location=device)
                                load_weights_dict = {k: v for k, v in weights_dict.items()
                                                     if model.state_dict()[k].numel() == v.numel()}
                                print(model.load_state_dict(load_weights_dict, strict=False))
                            else:
                                raise FileNotFoundError("not found weights file: {}".format(args.weights))

                        # 是否冻结权重
                        if args.freeze_layers:
                            for name, para in model.named_parameters():
                                # 除head外，其他权重全部冻结
                                if "head" not in name:
                                    para.requires_grad_(False)
                                else:
                                    print("training {}".format(name))

                        # pg = [p for p in model.parameters() if p.requires_grad]  # 保留需要学习的参数，并构成列表

                        if args.optimizer == "sgd":
                            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
                        elif args.optimizer == "Adam":
                            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay, amsgrad=True)
                        elif args.optimizer == "AdamW":
                            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay, amsgrad=True)
                        else:
                            print("没有该优化器")
                            exit()

                        # TODO
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                         factor=0.5, patience=20, min_lr=0.00001)
                        best_auroc = 0.0
                        epochAcc = 0
                        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
                        acc_list = []
                        auroc_dist_list = []
                        oscr_dist_list = []
                        for epoch in range(args.epochs):
                            # train
                            train_loss, train_acc = train_one_epoch(model=model,
                                                                    optimizer=optimizer,
                                                                    data_loader=train_dataloader,
                                                                    device=device,
                                                                    epoch=epoch,
                                                                    para_lambda=para_lambda
                                                                    )
                            tb_writer.add_scalars('lr', {tags[4]: optimizer.param_groups[0]["lr"]}, epoch)
                            scheduler.step(train_loss)  # 调整学习率
                            tb_writer.add_scalars('loss', {tags[0]: train_loss}, epoch)
                            tb_writer.add_scalars('acc', {tags[1]: train_acc}, epoch)

                            if epoch + 1 >= 131:
                                # validate
                                val_acc = evaluate(model=model,
                                                   data_loader=val_dataloader,
                                                   device=device,
                                                   epoch=epoch
                                                   )
                                # tb_writer.add_scalars('loss', {tags[2]: val_loss}, epoch)
                                tb_writer.add_scalars('acc', {tags[3]: val_acc}, epoch)
                                auroc_dist, oscr_dist = evaluate_auroc_dist(model=model,
                                                                            data_loader_known=known_dataloader,
                                                                            data_loader_unknown=unknown_dataloader,
                                                                            data_loader_known_oscr=known_dataloader_oscr,
                                                                            device=device
                                                                            )
                                acc_list.append(val_acc)
                                auroc_dist_list.append(auroc_dist)
                                oscr_dist_list.append(oscr_dist)

                                for param_group in optimizer.param_groups:
                                    print('epoch = ', epoch + 1, 'lr = ', param_group['lr'], "auroc_dist = ", auroc_dist)
                                for param_group in optimizer.param_groups:
                                    print('epoch = ', epoch + 1, 'lr = ', param_group['lr'], "oscr_dist = ", oscr_dist)

                                if epoch + 1 > args.epochs - 1:
                                    savefile = (args.save_path + backbone + '_' + loss_f +
                                                '_data' + str(data_number) + '_l2×' + str(weight_decay)
                                                + '_lil×' + str(para_lambda) + '.pth')
                                    torch.save(model.state_dict(), savefile)

                        acc_array = np.array(acc_list)
                        save_auroc_file = ('./txt/acc_' + backbone + '_' + loss_f +
                                           '_data' + str(data_number) + '_l2×' + str(weight_decay)
                                           + '_lil×' + str(para_lambda) + '.txt')
                        np.savetxt(save_auroc_file, acc_array)
                        auroc_dist_array = np.array(auroc_dist_list)
                        save_auroc_file = ('./txt/auroc_train_dist_' + backbone + '_' + loss_f +
                                           '_data' + str(data_number) + '_l2×' + str(weight_decay)
                                           + '_lil×' + str(para_lambda) + '.txt')
                        np.savetxt(save_auroc_file, auroc_dist_array)
                        oscr_dist_array = np.array(oscr_dist_list)
                        save_oscr_file = ('./txt/oscr_train_dist_' + backbone + '_' + loss_f +
                                          '_data' + str(data_number) + '_l2×' + str(weight_decay)
                                          + '_lil×' + str(para_lambda) + '.txt')
                        np.savetxt(save_oscr_file, oscr_dist_array)

                        print("Train finishing! The best_acc:{:.2f}%. EpochAcc:{}".format(
                            best_auroc * 100, epochAcc + 1))
                        print("batch-size:{},lr:{}||save_path:{}||optimizer:{}||weights:{}||T_max:{}||"
                              "weight_decay:{}||logs_path:{}""||save_path:{}||"
                              "label_smoothing:{}".format(args.batch_size, args.lr, args.save_path, args.optimizer,
                                                          args.weights, args.T_max, args.weight_decay,
                                                          args.logs_path, args.save_path, args.label_smoothing))

                        print('Start Tensorboard with "tensorboard --logdir={}", view at http://localhost:6006/'.
                              format(args.logs_path))


if __name__ == '__main__':
    """
    num_classes
    epochs
    batch-size
    lr
    save_path
    optimizer
    train_path
    val_path
    weights
    freeze-layers
    device
    T_max
    """

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--weights', type=str,
                        default='', help='initial weights path')
    # parser.add_argument('--weights', type=str,
    #                     default='./weight/alexnet_ring_data3_l2×0_mixup×0.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=None)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--val_size', type=int, default=128)
    parser.add_argument('--roc_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--T_max', type=int, default=20)
    parser.add_argument('--logs_path', type=str, default="./log/9_12")
    parser.add_argument('--save_path', type=str, default='./weight/')
    parser.add_argument('--backbone', type=str, default=['alexnet'])
    # parser.add_argument('--loss_f', type=str, default=['kpf', 'cac', 'rpl', 'arpl', 'slcpl', 'ring', 'softmax'])
    parser.add_argument('--loss_f', type=str, default=['gcpl'])
    parser.add_argument('--optimizer', type=str, default='AdamW')

    # Scenario1
    parser.add_argument('--train_path1', type=str, default="data/traindata_1.csv")
    parser.add_argument('--val_path1', type=str, default="data/testdata_1.csv")
    parser.add_argument('--known_path1', type=str, default="data/testdata_1.csv")
    parser.add_argument('--unknown_path1', type=str, default="data/testdata_unknown_1.csv")
    parser.add_argument('--class_indices_json_path1', type=str, default="./json/class_indices_1.json")
    parser.add_argument('--roc_indices_json_path1', type=str, default="./json/roc_indices_1.json")
    parser.add_argument('--num_classes1', type=int, default=4)

    # Scenario2
    parser.add_argument('--train_path2', type=str, default="data/traindata_2.csv")
    parser.add_argument('--val_path2', type=str, default="data/testdata_2.csv")
    parser.add_argument('--known_path2', type=str, default="data/testdata_2.csv")
    parser.add_argument('--unknown_path2', type=str, default="data/testdata_unknown_2.csv")
    parser.add_argument('--class_indices_json_path2', type=str, default="./json/class_indices_2.json")
    parser.add_argument('--roc_indices_json_path2', type=str, default="./json/roc_indices_2.json")
    parser.add_argument('--num_classes2', type=int, default=4)

    # Scenario3
    parser.add_argument('--train_path3', type=str, default="data/traindata_sim_5kc_12.csv")
    parser.add_argument('--val_path3', type=str, default="data/testdata_sim_5kc_15.csv")
    parser.add_argument('--known_path3', type=str, default="data/testdata_sim_5kc_15.csv")
    parser.add_argument('--unknown_path3', type=str, default="data/unknown_sim_5kc_15.csv")
    parser.add_argument('--class_indices_json_path3', type=str, default="./json/class_indices_3.json")
    parser.add_argument('--roc_indices_json_path3', type=str, default="./json/roc_indices_3.json")
    parser.add_argument('--num_classes3', type=int, default=5)

    # Scenario4
    parser.add_argument('--train_path4', type=str, default="data/traindata_sim_6kc_12.csv")
    parser.add_argument('--val_path4', type=str, default="data/testdata_sim_6kc_27.csv")
    parser.add_argument('--known_path4', type=str, default="data/testdata_sim_6kc_27.csv")
    parser.add_argument('--unknown_path4', type=str, default="data/unknown_sim_6kc_27.csv")
    parser.add_argument('--class_indices_json_path4', type=str, default="./json/class_indices_4.json")
    parser.add_argument('--roc_indices_json_path4', type=str, default="./json/roc_indices_4.json")
    parser.add_argument('--num_classes4', type=int, default=6)

    parser.add_argument('--data_number', type=float, default=[3, 4])
    parser.add_argument('--weight_decay', type=float, default=[0.05])
    parser.add_argument('--para_lambda_lil', type=float, default=[0])
    # parser.add_argument('--para_lambda_lil', type=float, default=[0, 0.1])
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    opt = parser.parse_args()

    setup_seed(42)
    main(opt)

    end = time.time()
    print("训练花费时间：{}秒".format(time.strftime("%H:%M:%S", time.gmtime(end - start))))
