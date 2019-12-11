import tensorflow as tf
import random
import pathlib
from sklearn.model_selection import train_test_split

class DataSetGenerator():
    def __init__(self, IMG_HEIGHT, IMG_WIDTH):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH

    def load_and_preprocess_image(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.IMG_HEIGHT, self.IMG_WIDTH])
        img /= 255.0  # normalize to [0,1] range
        return img

    def prepare_for_training(self, ds, cache=True):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
        else:
          ds = ds.cache()

        ds = ds.shuffle(buffer_size=self.SHUFFLE_SIZE)

        ds = ds.batch(self.BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def labeled_dataset(self, image_paths, labels):
        # a dataset that returns image paths
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        # for n, img_path in enumerate(path_ds.take(4)):
        #     print(n, img_path)

        # a dataset that returns images (loaded off disk, decoded, and preprocessed)
        image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        # for n, image in enumerate(image_ds.take(4)):
        #     print(n, image.shape)

        # a dataset that returns labels
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
        # for label in label_ds.take(4):
        #     print(self.label_names[label.numpy()])

        # a dataset that returns images and labels
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        # for img, label in image_label_ds.take(2):
        #     print(img.shape, self.label_names[label.numpy()])

        return image_label_ds

    def get_final_dataset(self, data_root, SHUFFLE_SIZE, BATCH_SIZE):
        self.SHUFFLE_SIZE =SHUFFLE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        # a dataset that returns image paths
        data_root = pathlib.Path(data_root)
        all_image_paths = list(data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths if str(path).lower().endswith("png") or
                                                                    str(path).lower().endswith("jpg")]
        random.shuffle(all_image_paths)
        print('There are %d images in this dataset.' % len(all_image_paths))

        self.label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        print('All categories in the dataset:', self.label_names)
        label_to_index = dict((name, index) for index, name in enumerate(self.label_names))
        all_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

        # separate dataset
        train_paths, test_paths, train_labels, test_labels = train_test_split(all_image_paths, all_labels)

        train_ds = self.labeled_dataset(train_paths, train_labels)
        train_ds = self.prepare_for_training(train_ds)

        test_ds = self.labeled_dataset(test_paths, test_labels)
        test_ds = self.prepare_for_training(test_ds)

        return train_ds, test_ds
