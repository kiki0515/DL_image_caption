import argparse, os
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
import torch.backends.cudnn as cudnn

import pickle
import numpy as np
from utils import Vocabulary, Custom_Flickr, collate_fn
from models import SimpleEncoderCNN, EncoderCNN, DecoderRNNwithAttention
from BLEU import bleu_eval


def get_parser():
    parser = argparse.ArgumentParser(description='Flickr8k Training')
    parser.add_argument('-batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('-hid_size', type=int, default=512,
                        help='hidden demension size')
    parser.add_argument('-attn_size', type=int, default=512,
                        help='attention demension size')
    parser.add_argument('-drop', type=float, default=0.5,
                        help='dropout percentage')
    parser.add_argument('-epoch', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('-fine_tune', type=bool, default=False,
                        help='whether to fine-tune the encoder or not')
    parser.add_argument('-lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--save', type=str, default=Path.cwd(),
                        help='directory to save logs and models.')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    return parser


def main():

    image_file = 'data_flickr8k/Images/'
    ann_file = 'data_flickr8k/Flickr8k.token.txt'

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_parser().parse_args()

    NUM_WORKERS = 2
    CROP_SIZE = 256
    NUM_PIXELS = 64
    ENCODER_SIZE = 2048
    ALPHA = 1.  # attention regularization parameter
    learning_rate = args.lr
    start_epoch = 0

    max_BLEU = 0

    vocab = pickle.load(open('vocab.p', 'rb'))

    train_transform = transforms.Compose([
            transforms.RandomCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.444, 0.421, 0.385),
                                 (0.285, 0.277, 0.286))])

    test_transform = transforms.Compose([
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.444, 0.421, 0.385),
                                 (0.285, 0.277, 0.286))])

    train_loader = torch.utils.data.DataLoader(
            dataset=Custom_Flickr(image_file,ann_file, vocab,  train=True, val = False, test=False, transform=train_transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
            dataset=Custom_Flickr(image_file,ann_file, vocab, train=False, val=True, test=False, transform=test_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(
        dataset=Custom_Flickr(image_file, ann_file, vocab, train=False, val=False, test=True, transform=test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn)

    # Initialize models
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNNwithAttention(vocab, args.hid_size, 1, args.attn_size, ENCODER_SIZE, NUM_PIXELS, dropout=args.drop).to(device)



    # Initialize optimization
    criterion = torch.nn.CrossEntropyLoss()
    #decoder.embed.weight.requires_grad = False
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            max_BLEU = checkpoint['max_BLEU']
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else: print("No checkpoint found at '{}'".format(args.resume))

    XEntropy = AverageMeter()
    PPL = AverageMeter()

    # Save
    if not args.resume:
        file = open(f'{args.save}/resuts.txt','a')
        file.write('Loss,PPL,BLEU \n')
        file.close()

    for epoch in range(start_epoch, 30):

        print('Epoch {}'.format(epoch+1))
        print('training...')
        for i, (images, captions, lengths) in enumerate(train_loader):

            if i%10 ==  0:
                print('[{}/{}]'.format(i,len(train_loader)))
                print(PPL.avg)

            # Batch to device
            images = images.to(device)
            captions = captions.to(device)
            lengths.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            encoder.train()
            decoder.train()

            features = encoder(images)
            predictions, attention_weights = decoder(features, captions, lengths)

            scores = pack_padded_sequence(predictions[:,:-1,:], lengths-2, batch_first=True)
            targets = pack_padded_sequence(captions[:,1:-1], lengths-2, batch_first=True)

            loss = criterion(scores.data, targets.data)
            loss += ALPHA * ((1. - attention_weights.sum(dim=1)) ** 2).mean()

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            XEntropy.update(loss.item(), len(lengths))
            PPL.update(np.exp(loss.item()), len(lengths))
        print('Train Perplexity = {}'.format(PPL.avg))

        if epoch+1 % 10 == 0:
            learning_rate /= 10
            for param_group in optimizer.param_groups: param_group['lr'] = learning_rate

        encoder.eval()
        decoder.eval()
        print('validating...')
        _, curr_BLEU = bleu_eval(encoder, decoder, val_loader, args.batch_size, device)
        print(curr_BLEU)
        #is_best = curr_BLEU > max_BLEU
        #max_BLEU = max(curr_BLEU, max_BLEU)
        save_checkpoint({
            'epoch': epoch + 1, 'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
            'max_BLEU': curr_BLEU, 'optimizer' : optimizer.state_dict(),
        }, True, args.save)

        print('Validation BLEU = {}'.format(curr_BLEU))

        # Save
        file = open(f'{args.save}/resuts.txt','a')
        file.write('{},{},{} \n'.format(XEntropy.avg,PPL.avg,curr_BLEU))
        file.close()

    checkpoint = torch.load(f'{args.save}/model_best.pth.tar')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    #decoder.embed.weight.requires_grad = True
    learning_rate = 0.001
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for epoch in range(start_epoch, args.epoch):
        print('Epoch {}'.format(epoch+1))
        print('training...')
        for i, (images, captions, lengths) in enumerate(train_loader):

            if i%10 ==  0:
                print('[{}/{}]'.format(i,len(train_loader)))
                print(PPL.avg)

            # Batch to device
            images = images.to(device)
            captions = captions.to(device)
            lengths.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            encoder.train()
            decoder.train()

            features = encoder(images)
            predictions, attention_weights = decoder(features, captions, lengths)

            scores = pack_padded_sequence(predictions[:,:-1,:], lengths-2, batch_first=True)
            targets = pack_padded_sequence(captions[:,1:-1], lengths-2, batch_first=True)

            loss = criterion(scores.data, targets.data)
            loss += ALPHA * ((1. - attention_weights.sum(dim=1)) ** 2).mean()

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            XEntropy.update(loss.item(), len(lengths))
            PPL.update(np.exp(loss.item()), len(lengths))
        print('Train Perplexity = {}'.format(PPL.avg))

        if epoch+1 % 5 == 0:
            learning_rate /= 10
            for param_group in optimizer.param_groups: param_group['lr'] = learning_rate

        encoder.eval()
        decoder.eval()
        print('validating...')
        print(bleu_eval(encoder, decoder, val_loader, args.batch_size, device)[0])
        print(bleu_eval(encoder, decoder, val_loader, args.batch_size, device)[1])
        print(bleu_eval(encoder, decoder, val_loader, args.batch_size, device)[2])

        curr_BLEU = bleu_eval(encoder, decoder, val_loader, args.batch_size, device)[0]
        is_best = curr_BLEU > max_BLEU
        max_BLEU = max(curr_BLEU, max_BLEU)
        save_checkpoint({
            'epoch': epoch + 1, 'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
            'max_BLEU': max_BLEU, 'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)

        print('Validation BLEU = {}'.format(curr_BLEU))

        # Save
        file = open(f'{args.save}/resuts.txt','a')
        file.write('{},{},{} \n'.format(XEntropy.avg,PPL.avg,curr_BLEU))
        file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, save, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{save}/model_best.pth.tar')


if __name__ == '__main__': main()
