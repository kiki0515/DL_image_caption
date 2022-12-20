from PIL import Image
import numpy as np
from utils import Vocabulary
from tqdm import tqdm
import nltk
from collections import Counter
import multiprocessing as mp
import pickle
import os


def get_splits(image_filepath, size=256, seed=2020):
    np.random.seed(seed)
    train_set = set()
    val_set = set()
    test_set = set()
    num_of_dropped = 0

    cases = [0, 1, 2]
    probs = [0.7, 0.1, 0.2]

    for filename in tqdm(os.listdir(image_filepath)):
        if filename[-3:] == 'jpg':
            img = np.array(Image.open(os.path.join(image_filepath, filename)))
            if img.shape[0] >= size and img.shape[1] >= size:
                selector = np.random.choice(cases, p=probs)
                if selector == 0:
                    train_set.add(filename)
                elif selector == 1:
                    val_set.add(filename)
                elif selector == 2:
                    test_set.add(filename)
            else:
                num_of_dropped += 1

    pickle.dump(train_set, open('train_set.p', 'wb'))
    pickle.dump(val_set, open('val_set.p', 'wb'))

    print('Droppped {} Images'.format(num_of_dropped))


def build_vocab(ann_file, threshold = 10):
    """Build a simple vocabulary wrapper."""
    punc_set = set([',',';',':','.','?','!','(',')','"','``'])
    counter = Counter()
    caption_list = []
    split = pickle.load(open('train_set.p', 'rb'))
    ann_file = os.path.expanduser(ann_file)
    with open(ann_file) as fh:
        for line in fh:
            img, caption = line.strip().split('\t')
            if img[:-2] in split:
                caption_list.append(caption)

    pool = mp.Pool(mp.cpu_count())
    tokens = pool.map(nltk.tokenize.word_tokenize, [caption.lower() for caption in tqdm(caption_list)])
    pool.close()
    tokens = [ item for elem in tokens for item in elem]
    tokens = [elem for elem in tokens if elem not in punc_set]
    counter = Counter(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    vocab.add_word('<break>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    pickle.dump(vocab, open('vocab.p', 'wb'))


if __name__ == '__main__':
    image_filepath = 'data_flickr8k/Images/'
    ann_file = 'data_flickr8k/Flickr8k.token.txt'
    get_splits(image_filepath)
    build_vocab(ann_file,threshold = 10)
