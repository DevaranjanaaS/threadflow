import os
import gzip
import numpy as np
import urllib.request

class MNISTLoader:
    URLS = {
        'train_img': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_lbl': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_img': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_lbl': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }

    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _download(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            # Using a more reliable mirror if Yann LeCun's site is down
            fallback_url = f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}"
            try:
                urllib.request.urlretrieve(self.URLS[filename.replace('.gz', '').replace('-', '_')], path)
            except:
                print(f"Primary URL failed, trying fallback...")
                urllib.request.urlretrieve(fallback_url, path)
        return path

    def _load_images(self, path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def _load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int64)

    def load(self):
        train_img_path = self._download('train-images-idx3-ubyte.gz')
        train_lbl_path = self._download('train-labels-idx1-ubyte.gz')
        test_img_path = self._download('t10k-images-idx3-ubyte.gz')
        test_lbl_path = self._download('t10k-labels-idx1-ubyte.gz')

        x_train = self._load_images(train_img_path)
        y_train = self._load_labels(train_lbl_path)
        x_test = self._load_images(test_img_path)
        y_test = self._load_labels(test_lbl_path)

        return (x_train, y_train), (x_test, y_test)

def get_batches(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    if shuffle:
        indices = np.random.permutation(n)
        x, y = x[indices], y[indices]
    
    for i in range(0, n, batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]
