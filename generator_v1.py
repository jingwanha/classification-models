import numpy as np
import keras
import cv2
import imgaug.augmenters as iaa

class BalencedDataGenerator(keras.utils.Sequence):
    
    def __init__(self, data, config, is_train=True):
        'Initialization'
        self.data = data
        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.n_classes = config.NUM_CLASS
        self.h = config.INPUT_SHAPE[0]
        self.w = config.INPUT_SHAPE[1]
        self.is_train = is_train
        self.unique_label = np.unique(data["label"])
        self.seq = iaa.Sequential([iaa.Fliplr(0.5),                      
                                   iaa.Affine(rotate=(-10,10)),           
                                   iaa.Multiply((0.8, 1.2)),
                                   iaa.PadToFixedSize(width=self.w+50, height=self.h+50),
                                   iaa.CropToFixedSize(self.h,self.w,'normal')
                                  ])
        
        self.on_epoch_end()
        
    def __len__(self):
        return int(len(self.data["image"]) / float(self.batch_size))

    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        if self.is_train:
            x, y = self.__data_generation(indexes, self.data, self.is_train)
            
#             x = self.seq.augment_images(x)
#             x = np.clip(x, 0.,1.)  # if images`s brightness max value is over 255, clip it.
        else:
            x, y = self.__data_generation(indexes, self.data, self.is_train)
        
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.array(list(range(self.data.shape[0])))
        if self.is_train == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes, data, is_train):
        'Generates data containing batch_size samples'
        
        if is_train:
            balenced_index = []

            while True:
                
                # print(balenced_index)

                if len(balenced_index) >= self.batch_size:

                    break

                for ul in self.unique_label:

                    ul_data = data[data["label"] == ul]
                    ul_index = ul_data.index
                    random_index = np.random.choice(ul_index,1)[0]
                    balenced_index.append(random_index)
                    
                    
            balenced_index = balenced_index[:self.batch_size]
                    
            X = np.empty((self.batch_size, self.h, self.w, 3),dtype=np.float32)
            y = np.empty((self.batch_size), dtype=int)

            for i, idx in enumerate(balenced_index) :
                # Read data
                image = cv2.imread(data["image"][idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = plt.imread(data["image"][idx])

                label = data["label"][idx]
                
                # image processing
                image = remove_padding(image)
                image = crop_or_pad(image, [self.h,self.w])

                # append batch
                X[i,] = image
                y[i,] = int(label)
                
        else:
            
            X = np.empty((self.batch_size, self.h, self.w, 3), dtype=np.float32)
            y = np.empty((self.batch_size), dtype=int)

            for i, idx in enumerate(indexes) :
                # Read data
                image = cv2.imread(data["image"][idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = plt.imread(data["image"][idx])

                label = data["label"][idx]
                # image processing

                image = remove_padding(image)
                image = crop_or_pad(image, [self.h,self.w])

                # append batch
                X[i,] = image
                y[i,] = int(label)
        return X, y
    

def remove_padding(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def show_im(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


def crop_or_pad(image, size=(299, 299)) :
        
    image = image.astype(np.float32)
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    image_max = max(h, w)
    scale = float(min(size)) / image_max

    image = cv2.resize(image, (int(w * scale), int(h * scale)))

    h, w = image.shape[:2]
    top_pad = (size[1] - h) // 2
    bottom_pad = size[1] - h - top_pad
    left_pad = (size[0] - w) // 2
    right_pad = size[0] - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    
    # Fix image normalization
    if np.nanmax(image) > 1 :
        image = np.divide(image, 255.)

    return image