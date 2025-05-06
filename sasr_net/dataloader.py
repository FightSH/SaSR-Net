from typing import Optional, Sequence, List, Any, Callable, Dict
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _BaseDataLoaderIter, _collate_fn_t, _worker_init_fn_t
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
from munch import munchify
import time
import random
import torch.nn.functional as F

from typing import *
from collections import OrderedDict

T = TypeVar('T')
S = TypeVar('S')


class LRU_cache(Generic[T, S]):
    def __init__(self, max_size: Optional[int] = None) -> None:
        self.cache: OrderedDict[T, S] = OrderedDict()
        self.count: int = 0
        self.max_size: int = max_size

    def setdefault(self, key: T, _default_value: Optional[S] = None) -> S:
        if key in self.cache:
            return self.cache[key]
        self.cache[key] = _default_value
        self.count += 1
        if self.max_size is not None and self.count > self.max_size:
            self.cache.popitem(last=False)
        return _default_value

    def __len__(self) -> int:
        return self.count


def func_ids_to_multinomial(categories):
    id_to_idx = {id: index for index, id in enumerate(categories)}

    def ids_to_multinomial(id):
        """ label encoding
        Returns:
        1d array, multimonial representation, e.g. [1,0,1,0,0,...]
        """

        return id_to_idx[id]

    return ids_to_multinomial


class SaSRDataset(Dataset):

    def __init__(self, label, audio_dir, video_res14x14_dir, transform=None, mode_flag='train'):
        """
        初始化数据集类，用于加载音频、视觉特征和问题数据。

        参数:
            label (str): 标签文件路径，包含问题和答案信息。
            audio_dir (str): 音频特征文件目录路径。
            video_res14x14_dir (str): 视觉特征文件目录路径。
            transform (callable, optional): 数据预处理函数，默认为None。
            mode_flag (str): 数据集模式标志，'train' 表示训练模式，其他值表示非训练模式。
        """
        self.train: bool = mode_flag == "train"  # 判断是否为训练模式

        # 加载样本数据
        samples = json.load(open('./data/json/avqa-train.json', 'r'))

        # 构建问题词汇表和答案词汇表
        ques_vocab = ['<pad>']  # 初始化问题词汇表，包含填充符
        ans_vocab = []  # 初始化答案词汇表
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')  # 分割问题内容
            question[-1] = question[-1][:-1]  # 移除问题末尾的问号

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:  # 替换问题中的模板标记为实际值
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1

            for wd in question:  # 将问题中的单词添加到词汇表
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:  # 将答案添加到答案词汇表
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab  # 问题词汇表
        self.ans_vocab = ans_vocab  # 答案词汇表
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}  # 词汇到索引的映射

        # 加载完整数据集
        self.samples = json.load(open(label, 'r'))
        self.max_len = 14  # 问题的最大长度

        self.audio_dir = audio_dir  # 音频特征目录
        self.video_res14x14_dir = video_res14x14_dir  # 视觉特征目录
        self.transform = transform  # 数据预处理函数

        # 构建视频列表
        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list  # 视频列表
        self.video_len = 60 * len(video_list)  # 视频帧总数
        self.frame_ids: np.ndarray[int] = np.arange(self.video_len)  # 帧索引数组

        # 初始化LRU缓存，用于存储音频和视觉特征
        self.audio_data: LRU_cache[str, np.ndarray[Any]] = LRU_cache(max_size=None)
        self.visual_data: LRU_cache[str, np.ndarray[Any]] = LRU_cache(max_size=None)

        # 初始化答案到索引的映射函数
        self.ids_to_multinomial = func_ids_to_multinomial(self.ans_vocab)
        self.items: List[str] = ['cello', 'congas', 'pipa', 'ukulele', 'piano', 'accordion', 'clarinet', 'guzheng',
                                 'saxophone', 'drum', 'violin', 'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo',
                                 'electric_bass', 'flute', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取数据集中的单个样本

        参数:
            idx (int): 样本索引

        返回:
            dict: 包含音频、视觉正样本、视觉负样本、问题、标签和物品特征的字典
        """
        # 获取对应索引的样本数据
        sample = self.samples[idx]
        name = sample['video_id']  # 获取视频ID

        # 根据训练模式决定是否获取物品信息
        if self.train:
            items: Dict[str, str] = sample["items"]  # 训练模式下获取物品信息
        else:
            items = {}  # 非训练模式下使用空字典

        # 加载音频特征，使用LRU缓存避免重复加载
        audio = self.audio_data.setdefault(name, np.load(os.path.join(self.audio_dir, name + '.npy'), mmap_mode='r'))
        audio = audio[::6, :]  # 对音频特征进行下采样，每6帧取1帧

        # 加载视觉正样本特征，同样使用缓存
        visual_posi = self.visual_data.setdefault(name, np.load(os.path.join(self.video_res14x14_dir, name + '.npy'),
                                                                mmap_mode='r'))
        visual_posi = visual_posi[::6, :]  # 对视觉特征也进行相同的下采样
        video_idx = self.video_list.index(name)  # 获取当前视频在视频列表中的索引

        # 为每一帧生成负样本帧的ID，确保负样本来自不同的视频
        neg_frame_ids: List[int] = [random_int(0, self.video_len - 1, lambda x: x // 60 != video_idx) for _ in
                                    range(visual_posi.shape[0])]

        # 用于存储视觉负样本的列表
        visual_nega_list: List[np.ndarray[int]] = []

        # 为每一帧构建对应的负样本
        for i in range(visual_posi.shape[0]):
            neg_frame_id: int = neg_frame_ids[i]  # 获取负样本帧ID

            neg_video_id: int = neg_frame_id // 60  # 计算负样本视频ID（每个视频有60帧）
            neg_frame_flag: int = neg_frame_id % 60  # 计算负样本在视频中的帧位置

            neg_video_name: str = self.video_list[neg_video_id]  # 获取负样本视频名称

            # 加载负样本视频特征
            visual_nega_out_res18 = self.visual_data.setdefault(neg_video_name, np.load(
                os.path.join(self.video_res14x14_dir, neg_video_name + '.npy'), mmap_mode='r'))
            visual_nega_list.append(visual_nega_out_res18[neg_frame_flag, :, :, :])  # 添加特定帧的特征到列表

        # 将负样本列表堆叠成一个张量
        visual_nega: Any = np.stack(visual_nega_list, axis=0)
        visual_nega: Any = torch.from_numpy(visual_nega)  # 转换为PyTorch张量

        # 处理问题文本
        question_id = sample['question_id']  # 获取问题ID
        question = sample['question_content'].rstrip().split(' ')  # 分词
        question[-1] = question[-1][:-1]  # 移除最后一个词的问号

        # 替换问题中的模板标记为实际值
        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:  # 检测模板标记
                question[pos] = ast.literal_eval(sample['templ_values'])[p]  # 替换为实际值
                p += 1

        # 如果问题长度不足，用<pad>填充到指定长度
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')

        # 将单词转换为索引
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)  # 转换为PyTorch张量

        # 处理答案
        answer = sample['anser']  # 获取答案
        label = self.ids_to_multinomial(answer)  # 将答案转换为索引
        label = torch.from_numpy(np.array(label)).long()  # 转换为PyTorch张量

        # 构建最终的样本字典
        sample = {'audio': audio, 'visual_posi': visual_posi, 'visual_nega': visual_nega,
                  'question': ques, 'label': label, 'items': self.items_to_embed(items)}

        # 如果存在转换函数，应用转换
        if self.transform:
            sample = self.transform(sample)

        return sample

    def items_to_embed(self, items: Dict[str, str]) -> np.ndarray:
        res: np.ndarray = np.zeros(len(self.items))
        for i, item in enumerate(self.items):
            res[i] = items.get(item, 0)
        return res


def random_int(min_value: int = 0, max_value: int = 10000, filter_key: Callable[[int], bool] = lambda _: True) -> int:
    while True:
        i = random.randint(0, max_value)
        if filter_key(i):
            return i


class ToTensor:

    def __call__(self, sample):
        audio = sample['audio']
        visual_posi = sample['visual_posi']
        visual_nega = sample['visual_nega']
        question = sample['question']
        label = sample['label']
        items = sample["items"]
        # label = F.one_hot(sample['label'], num_classes=42)

        return {
            'audio': torch.from_numpy(audio),
            'visual_posi': visual_posi,
            'visual_nega': visual_nega,
            'question': question,
            'label': label,
            "items": torch.from_numpy(items)}
