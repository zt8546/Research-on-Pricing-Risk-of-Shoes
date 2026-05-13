import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPU devices: {gpus}")
assert len(gpus) > 0, "No GPU was detected. Training on CPU will be extremely slow."

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
BRANDS = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 20
EPOCHS_FINETUNE = 60


class SmartAdaptiveLR(Callback):
    def __init__(
        self,
        init_lr,
        max_lr,
        min_lr,
        up_factor=1.2,
        down_factor=0.5,
        patience=2,
        down_patience=2,
        lr_log=None,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.up_factor = up_factor
        self.down_factor = down_factor
        self.patience = patience
        self.down_patience = down_patience
        self.lr_log = lr_log if lr_log is not None else []
        self._best_acc = -float('inf')
        self._best_loss = float('inf')
        self._good_streak = 0
        self._bad_streak = 0

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.init_lr)
        self._best_acc = -float('inf')
        self._best_loss = float('inf')
        self._good_streak = 0
        self._bad_streak = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        if val_acc is None or val_loss is None:
            return

        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        acc_improved = val_acc > self._best_acc + 1e-4
        loss_improved = val_loss < self._best_loss - 1e-4

        if acc_improved and loss_improved:
            self._best_acc = val_acc
            self._best_loss = val_loss
            self._good_streak += 1
            self._bad_streak = 0
            if self._good_streak >= self.patience:
                new_lr = min(current_lr * self.up_factor, self.max_lr)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                self._good_streak = 0
                status = f"continuous improvement, LR raised from {current_lr:.2e} to {new_lr:.2e}"
            else:
                new_lr = current_lr
                status = f"improving ({self._good_streak}/{self.patience}), LR stays at {current_lr:.2e}"
        else:
            self._good_streak = 0
            self._bad_streak += 1
            if self._bad_streak >= self.down_patience:
                new_lr = max(current_lr * self.down_factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                self._bad_streak = 0
                status = f"no improvement, LR lowered from {current_lr:.2e} to {new_lr:.2e}"
            else:
                new_lr = current_lr
                status = f"no improvement ({self._bad_streak}/{self.down_patience}), LR stays at {current_lr:.2e}"

        self.lr_log.append(new_lr)
        print(
            f"  [SmartAdaptiveLR] Epoch {epoch + 1}: acc={val_acc:.4f} "
            f"loss={val_loss:.4f} | {status}"
        )


def make_callbacks(checkpoint_name, init_lr, max_lr, min_lr, lr_log):
    return [
        SmartAdaptiveLR(
            init_lr=init_lr,
            max_lr=max_lr,
            min_lr=min_lr,
            up_factor=1.2,
            down_factor=0.5,
            patience=2,
            down_patience=2,
            lr_log=lr_log,
        ),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_name, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]


def plot_training_results(history, stage1_end):
    print("--- Plotting training curves ---")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    def add_boundary(ax):
        ax.axvline(x=stage1_end - 1, color='gray', linestyle='--', alpha=0.7, label='Fine-tune Start')

    axes[0, 0].plot(history['accuracy'], label='Train Acc', color='royalblue')
    axes[0, 0].plot(history['val_accuracy'], label='Val Acc', color='tomato')
    add_boundary(axes[0, 0])
    axes[0, 0].set_title('Accuracy Trend')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['loss'], label='Train Loss', color='royalblue')
    axes[0, 1].plot(history['val_loss'], label='Val Loss', color='tomato')
    add_boundary(axes[0, 1])
    axes[0, 1].set_title('Loss Trend')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['precision'], label='Train Precision', color='royalblue')
    axes[1, 0].plot(history['val_precision'], label='Val Precision', color='tomato')
    add_boundary(axes[1, 0])
    axes[1, 0].set_title('Precision Trend (macro)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['recall'], label='Train Recall', color='royalblue')
    axes[1, 1].plot(history['val_recall'], label='Val Recall', color='tomato')
    add_boundary(axes[1, 1])
    axes[1, 1].set_title('Recall Trend (macro)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, 'training_history.png'))
    plt.show()


def merge_history(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged


def plot_confusion_matrix(model, datagen):
    print("--- Generating confusion matrix ---")
    val_gen_eval = datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=BRANDS,
        shuffle=False,
    )
    val_gen_eval.reset()
    preds = model.predict(val_gen_eval)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen_eval.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=BRANDS, yticklabels=BRANDS)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, 'confusion_matrix.png'))
    plt.show()


def start_training():
    print(f"--- Starting training, classes: {BRANDS} ---")

    datagen = ImageDataGenerator()

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=BRANDS,
    )
    val_gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=BRANDS,
    )

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.6),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(BRANDS), activation='softmax'),
        ]
    )

    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]

    print(f"\n[Stage 1] Training classifier head for up to {EPOCHS_FROZEN} epochs...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=metrics,
    )

    lr_log_stage1 = []
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        callbacks=make_callbacks(
            checkpoint_name=os.path.join(CURRENT_DIR, 'best_stage1.h5'),
            init_lr=1e-3,
            max_lr=5e-3,
            min_lr=1e-6,
            lr_log=lr_log_stage1,
        ),
    )

    print(f"\n[Stage 2] Unfreezing the last 10 EfficientNetB0 layers for up to {EPOCHS_FINETUNE} epochs...")
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=metrics,
    )

    lr_log_stage2 = []
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=make_callbacks(
            checkpoint_name=os.path.join(CURRENT_DIR, 'best_brand_model.h5'),
            init_lr=1e-4,
            max_lr=5e-4,
            min_lr=1e-7,
            lr_log=lr_log_stage2,
        ),
    )

    print("\nTraining finished. Best model saved as best_brand_model.h5")

    history = merge_history(history1, history2)
    stage1_end = len(history1.history['accuracy'])
    plot_training_results(history, stage1_end)
    plot_confusion_matrix(model, datagen)


if __name__ == "__main__":
    start_training()
