import os
from PIL import Image
import numpy as np

def save_data(videos_dir, save_dir):
    for img_dir in sorted(os.listdir(videos_dir)):
        _32_imgs = []
        _64_imgs = []
        for root_dir in sorted(os.listdir(videos_dir+img_dir)):
            _32_img = []
            _64_img = []
            for img in sorted(os.listdir(videos_dir+img_dir+'/'+root_dir)):
                if img.endswith(".jpeg") or img.endswith(".png"):
                    image = Image.open(os.path.join(videos_dir+img_dir+'/'+root_dir, img))
                    im32 = image.resize((32,32), Image.ANTIALIAS)
                    im64 = image.resize((64,64), Image.ANTIALIAS)
                    _32_img.append(np.asarray(im32))
                    _64_img.append(np.asarray(im64))
            _32_imgs.append(np.asarray(_32_img))
            _64_imgs.append(np.asarray(_64_img))
        X_32 = np.asarray(_32_imgs)
        X_64 = np.asarray(_64_imgs)
        np.save(save_dir+"32/"+img_dir, X_32)
        np.save(save_dir+"64/"+img_dir, X_64)
        print("Saved "+img_dir+" videos with shape ", X_64.shape)

def load_data(path):
    X = np.load(path)
    return X

if __name__ == '__main__':
    save_data(videos_dir=os.getcwd()+'/videos/', save_dir=os.getcwd()+'/data/')
    # save_data(img_dir=os.getcwd()+'/../videos/lstm_dataset/labels/', name='lstm_labels')
    # X = load_data(path=os.getcwd()+'/../videos/lstm_dataset/features/lstm_features.npy')
    # X = load_data(path=os.getcwd()+'/../videos/lstm_dataset/labels/lstm_labels.npy')    
    # print(X.shape)