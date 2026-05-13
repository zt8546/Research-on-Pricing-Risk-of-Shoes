import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sns


class DatasetHandler:
    def __init__(self, dataset_dir, train_dir, test_dir, val_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

    def get_image_dataset_from_directory(self, dir_name):
        dir_path = os.path.join(self.dataset_dir, dir_name)
        return tf.keras.utils.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            color_mode='rgb',
            seed=42,
            batch_size=64,
            image_size=(128, 128),
        )

    def load_split_data(self):
        train_data = self.get_image_dataset_from_directory(self.train_dir)
        test_data = self.get_image_dataset_from_directory(self.test_dir)
        val_data = self.get_image_dataset_from_directory(self.val_dir)
        return train_data, test_data, val_data


class DeepfakeDetectorModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(128, 128, 3)))
        model.add(layers.Rescaling(1.0 / 127, name='rescaling'))
        model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def compile_model(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

    def train_model(self, train_data, val_data, epochs):
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)
        model_checkpoint_callback = ModelCheckpoint(
            'deepfake_detector_model_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        )
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[early_stopping_callback, reduce_lr_callback, model_checkpoint_callback],
        )

    def evaluate_model(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, test_data):
        y_true, y_pred = [], []
        for images, labels in test_data:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend((preds.squeeze() > 0.5).astype(int))
        return np.array(y_true), np.array(y_pred)

    def save_model(self, path):
        self.model.save(path)


class Visualizer:
    @staticmethod
    def plot_training_history(history, save_path='training_history.png'):
        h = history.history
        precision_key = next(k for k in h if k.startswith('precision') and not k.startswith('val_'))
        recall_key = next(k for k in h if k.startswith('recall') and not k.startswith('val_'))
        val_precision_key = next(k for k in h if k.startswith('val_precision'))
        val_recall_key = next(k for k in h if k.startswith('val_recall'))

        metrics = [
            ('Accuracy Trend', 'Accuracy', 'accuracy', 'val_accuracy', 'Train Acc', 'Val Acc'),
            ('Loss Trend', 'Loss', 'loss', 'val_loss', 'Train Loss', 'Val Loss'),
            ('Precision Trend', 'Precision', precision_key, val_precision_key, 'Train Precision', 'Val Precision'),
            ('Recall Trend', 'Recall', recall_key, val_recall_key, 'Train Recall', 'Val Recall'),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for ax, (title, ylabel, train_key, val_key, train_label, val_label) in zip(axes, metrics):
            ax.plot(h[train_key], label=train_label, color='steelblue')
            ax.plot(h[val_key], label=val_label, color='tomato')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'Training history plot saved to {save_path}')

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'Confusion matrix saved to {save_path}')


class TrainModel:
    def __init__(self, dataset_dir, train_dir, test_dir, val_dir):
        self.dataset_handler = DatasetHandler(dataset_dir, train_dir, test_dir, val_dir)

    def run_training(self, learning_rate=0.0001, epochs=50):
        train_data, test_data, val_data = self.dataset_handler.load_split_data()
        class_names = train_data.class_names
        print(f'Class names: {class_names}')

        model = DeepfakeDetectorModel()
        model.compile_model(learning_rate)
        history = model.train_model(train_data, val_data, epochs)

        evaluation_metrics = model.evaluate_model(test_data)
        model.save_model('deepfake_detector_model.keras')
        Visualizer.plot_training_history(history, save_path='training_history.png')

        y_true, y_pred = model.predict(test_data)
        Visualizer.plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=class_names,
            save_path='confusion_matrix.png',
        )
        return history, evaluation_metrics


if __name__ == '__main__':
    dataset_dir = './Dataset'
    train_dir = 'Train'
    test_dir = 'Test'
    val_dir = 'Validation'

    trainer = TrainModel(
        dataset_dir=dataset_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        val_dir=val_dir,
    )

    history, evaluation_metrics = trainer.run_training(learning_rate=0.0001, epochs=50)
    print('Evaluation metrics:', evaluation_metrics)
