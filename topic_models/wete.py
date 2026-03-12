import torch
import numpy as np
import argparse
from topic_models.WeTe.dataloader import dataloader
from topic_models.WeTe.model import WeTe
from topic_models.WeTe.Trainer import Trainer


parser = argparse.ArgumentParser(description='WeTe')
parser.add_argument("--name", type=str, default="WeTe")
parser.add_argument("--dataset", type=str, default='20News')
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument("--eval_step", default=10, type=int)
parser.add_argument('--batch_size', type=int, default=500)

parser.add_argument('--n_topic_words', default=10, type=int)
parser.add_argument('--beta', type=float, default=0.5, help='balance coefficient for bidirectional transport cost (default: 0.5)')
parser.add_argument('--epsilon', type=float, default=1.0, help='trade-off between transport cost and likelihood (default: 1.0)')
parser.add_argument('--device', type=str, default="0", help='which device for training: 0, 1, 2, 3 (GPU) or cpu')
parser.add_argument('--init_alpha', type=bool, default=True, help='Using K-means to initialise topic embeddings or not, setting to True will make faster and better performance.')
parser.add_argument('--train_word', type=bool, default=True, help='Finetuning word embedding or not, seting to True will make better performance.')
parser.add_argument('--glove', type=str, default="topic_models/WeTe/glove.6B.100d.txt", help='load pretrained word embedding')
parser.add_argument('--embedding_dim', type=int, default=100, metavar='N')
args = parser.parse_args()
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
args.device = device

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, voc = dataloader(dataname=args.dataset, mode='train', batch_size=args.batch_size)
    test_loader, _ = dataloader(dataname=args.dataset, mode='test', batch_size=args.batch_size)
    args.vocsize = len(voc)
    print(f'=============================   Setting   =============================\n {args}')
    print(f'len train: {len(train_loader)}, len test: {len(test_loader)}, voc_len: {len(voc)}')

    model = WeTe(args, voc=voc)
    trainer = Trainer(args, model, voc=voc)
    trainer.train(train_loader, test_loader)