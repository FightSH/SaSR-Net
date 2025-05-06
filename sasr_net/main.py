from __future__ import print_function
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import *
from models import SaSR_Net
import ast
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import warnings
from datetime import datetime
import time
import logging

from loss import LossAVMatch, LossAVQA, LossCorrelation, LossSemantic

logging.basicConfig(level=logging.INFO)
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')
torch.set_printoptions(threshold=np.inf, edgeitems=120, linewidth=120)
writer = SummaryWriter('runs/net_avst/' + TIMESTAMP)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

logging.info(
    "\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")


def train(args, model, train_loader, optimizer, epoch):
    """
    训练模型的函数

    参数:
        args: 命令行参数对象
        model: 要训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前训练轮次
    """
    # 设置模型为训练模式
    model.train()

    # 初始化各种损失函数
    criterion_av_match = LossAVMatch()  # 音视频匹配损失
    criterion_avqa = LossAVQA()  # 音视频问答损失
    criterion_correlation = LossCorrelation(num_class=22)  # 相关性损失
    criterion_semantic = LossSemantic()  # 语义损失

    for batch_idx, sample in enumerate(train_loader):
        # 将数据移至GPU
        audio, visual_posi, visual_nega, target, question, items = sample['audio'].to('cuda'), sample['visual_posi'].to(
            'cuda'), sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), \
            sample["items"].to("cuda")

        # 清除梯度
        optimizer.zero_grad()

        # 前向传播，获取模型输出
        out_qa, out_match_posi, out_match_nega, av_cls_prob, v_prob, a_prob, _ = model(
            audio, visual_posi, visual_nega, question)

        # 计算各个损失项
        loss_match = criterion_av_match(out_match_posi, out_match_nega)  # 音视频匹配损失
        loss_cor = criterion_correlation(av_cls_prob)  # 相关性损失
        loss_semantic_v = criterion_semantic(v_prob, items)  # 视觉语义损失
        loss_semantic_a = criterion_semantic(a_prob, items)  # 音频语义损失
        loss_qa = criterion_avqa(out_qa, target)  # 问答损失

        # 合并语义损失
        loss_semantic = loss_semantic_v + loss_semantic_a

        # 计算总损失（带权重）
        loss = loss_qa + 0.5 * loss_match + 0.5 * loss_semantic + 0.5 * loss_cor

        # 记录各损失值到TensorBoard
        writer.add_scalar('run/loss_match', loss_match.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/loss_qa', loss_qa.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/loss_cor', loss_cor.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/loss_semantic', loss_semantic.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/loss_semantic_v', loss_semantic_v.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/loss_semantic_a', loss_semantic_a.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/Loss_all', loss.item(),
                          epoch * len(train_loader) + batch_idx)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 定期打印训练信息
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        # break  # 注意：这里的break会导致每个epoch只处理一个batch


def eval(model, val_loader, epoch):
    model.eval()
    total_qa = 0
    correct_qa = 0
    with torch.no_grad():
        for _, sample in enumerate(val_loader):
            audio, visual_posi, visual_nega, target, question, _ = sample['audio'].to('cuda'), sample['visual_posi'].to(
                'cuda'), sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), \
            sample["items"].to("cuda")

            out_qa, _, _, _, _, _, _ = model(
                audio, visual_posi, visual_nega, question)

            _, predicted = torch.max(out_qa.data, 1)
            total_qa += out_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    logging.info('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
    writer.add_scalar('metri_qa', 100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa


def test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('./data/json/avqa-test.json', 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, visual_posi, visual_nega, target, question, items = sample['audio'].to('cuda'), sample[
                'visual_posi'].to(
                'cuda'), sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), \
            sample["items"].to("cuda")

            preds_qa, _, _, _, _, _, _ = model(
                audio, visual_posi, visual_nega, question)
            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]
            type = ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())

    logging.info('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count) / len(A_count)))
    logging.info('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    logging.info('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    logging.info('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    logging.info('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    logging.info('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    logging.info('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    logging.info('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    logging.info('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    logging.info('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    logging.info('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    logging.info('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc) + sum(AV_ext) + sum(AV_temp)
                   + sum(AV_cmp)) / (len(AV_count) + len(AV_loc) + len(AV_ext) + len(AV_temp) + len(AV_cmp))))

    logging.info('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default='./dataset/avqa-features/feats/vggish', help="audio dir")
    parser.add_argument(
        "--video_res14x14_dir", type=str, default='./dataset/avqa-features/visual_14x14', help="res14x14 dir")

    parser.add_argument(
        "--label_train", type=str, default="./data/json/avqa-train-updated.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="./data/json/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="./data/json/avqa-test.json", help="test csv file")
    parser.add_argument(
        "--label_visualization", type=str, default="./data/json/avqa-val_real.json", help="visualization csv file")
    parser.add_argument(
        '--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        "--epochs", type=int, default=80, metavar="N", help="number of epochs to train (default: 60)")
    parser.add_argument(
        "--lr", type=float, default=1e-4, metavar="LR", help="learning rate (default: 3e-4)")
    parser.add_argument(
        "--model", type=str, default="sasr_net", help="with model to use")
    parser.add_argument(
        "--mode", type=str, default="train", help="with mode to use")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval", type=int, default=50, metavar="N",
        help="how many batches to wait before logging training status")
    parser.add_argument(
        "--model_save_dir", type=str, default="checkpoints/sasr_net/", help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default="sasr_net", help="save model name")
    parser.add_argument(
        "--gpu", type=str, default="0", help="gpu device number")
    parser.add_argument(
        "--pretrained_path", type=str, default="./pretrained/avst.pt", help="pretrained model that will be used"
    )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.model == 'sasr_net':
        model = SaSR_Net()
        model = model.to('cuda')
        # model = nn.DataParallel(model)
    else:
        raise ('not recognized')

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if args.mode == 'train':
        train_dataset = SaSRDataset(label=args.label_train, audio_dir=args.audio_dir,
                                    video_res14x14_dir=args.video_res14x14_dir,
                                    transform=transforms.Compose([ToTensor()]), mode_flag='train')
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataset = SaSRDataset(label=args.label_val, audio_dir=args.audio_dir,
                                  video_res14x14_dir=args.video_res14x14_dir,
                                  transform=transforms.Compose([ToTensor()]), mode_flag='val')
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        logging.info("Train Dataset", len(train_dataset))
        logging.info("Val Dataset", len(val_dataset))

        # ===================================== load pretrained model ===============================================
        # concat model
        pretrained_path = args.pretrained_path
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                checkpoint = torch.load(pretrained_path)
            except:
                checkpoint = {}
        else:
            checkpoint = {}
        logging.info(
            "\n-------------- loading pretrained models --------------")
        model_dict = model.state_dict()

        for key, value in checkpoint.items():
            potential_key = '.'.join(key.split('.')[1:])
            if potential_key in model_dict:
                model_dict[potential_key] = value
                logging.info("Successfully load layer {potential_key}.")
        model.load_state_dict(model_dict)

        logging.info("\n-------------- load pretrained models --------------")

        # ===================================== load pretrained model ===============================================

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=16, gamma=0.3)
        best_acc = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, epoch=epoch)
            scheduler.step(epoch)
            logging.info(
                f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            acc = eval(model, val_loader, epoch)
            if acc >= best_acc:
                best_acc = acc
                model_name: str = os.path.join(args.model_save_dir,
                                               args.checkpoint + f"_{TIMESTAMP.replace('/', '-')}std.pt")
                torch.save(model.state_dict(), model_name)
                logging.info(
                    f"Checkpoint epoch {epoch} acc {acc} has been saved, file name: {model_name}.")

    else:
        test_dataset = SaSRDataset(label=args.label_test, audio_dir=args.audio_dir,
                                   video_res14x14_dir=args.video_res14x14_dir,
                                   transform=transforms.Compose([ToTensor()]), mode_flag='test')
        logging.debug(test_dataset.__len__())
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        model.load_state_dict(torch.load(
            os.path.join(args.model_save_dir, args.checkpoint + ".pt")))
        test(model, test_loader)


if __name__ == '__main__':
    main()
