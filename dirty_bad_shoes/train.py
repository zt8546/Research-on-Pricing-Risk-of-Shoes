import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_train_val_dirs(start_dir):
    candidates = [
        start_dir,
        os.path.join(start_dir, 'dataset'),
        os.path.abspath(os.path.join(start_dir, '..', 'dataset')),
        os.path.abspath(os.path.join(start_dir, '..', '..', 'dataset')),
    ]
    for candidate in candidates:
        train_dir = os.path.join(candidate, 'train')
        val_dir = os.path.join(candidate, 'val')
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            return train_dir, val_dir
    raise FileNotFoundError(
        f"Cannot find train/val directories. Make sure the dataset is fully split.\n"
        f"Tried root directories: {candidates}"
    )


TRAIN_DIR, VAL_DIR = find_train_val_dirs(CURRENT_DIR)
CLASSES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 20
EPOCHS_FINETUNE = 60
UNFREEZE_LAYERS = 4
SEED = 42


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
        self._prev_acc = None
        self._prev_loss = None
        self._good_streak = 0
        self._bad_streak = 0

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.init_lr)
        self._prev_acc = None
        self._prev_loss = None
        self._good_streak = 0
        self._bad_streak = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        if val_acc is None or val_loss is None:
            return

        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        if self._prev_acc is None:
            self._prev_acc = val_acc
            self._prev_loss = val_loss
            self.lr_log.append(current_lr)
            print(
                f"  [SmartAdaptiveLR] Epoch {epoch + 1}: acc={val_acc:.4f} "
                f"loss={val_loss:.4f} | First round, learning rate remains {current_lr:.2e}"
            )
            return

        acc_delta = val_acc - self._prev_acc
        loss_delta = self._prev_loss - val_loss
        acc_better = acc_delta > 0
        loss_better = loss_delta > 0

        if acc_better and loss_better:
            is_improved = True
            reason = f"acc improved by {acc_delta:+.4f}, loss improved by {loss_delta:+.4f}"
        elif (not acc_better) and (not loss_better):
            is_improved = False
            reason = f"acc dropped by {acc_delta:+.4f}, loss worsened by {-loss_delta:+.4f}"
        else:
            improve_mag = acc_delta if acc_better else loss_delta
            worsen_mag = -loss_delta if acc_better else -acc_delta
            is_improved = improve_mag > worsen_mag
            tag = "improvement dominates" if is_improved else "degradation dominates"
            reason = (
                f"mixed signal: acc delta={acc_delta:+.4f}, loss delta={-loss_delta:+.4f} "
                f"({tag}: {improve_mag:.4f} vs {worsen_mag:.4f})"
            )

        self._prev_acc = val_acc
        self._prev_loss = val_loss

        if is_improved:
            self._good_streak += 1
            self._bad_streak = 0
            if self._good_streak >= self.patience:
                new_lr = min(current_lr * self.up_factor, self.max_lr)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                self._good_streak = 0
                status = f"raised LR from {current_lr:.2e} to {new_lr:.2e}"
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
                status = f"lowered LR from {current_lr:.2e} to {new_lr:.2e}"
            else:
                new_lr = current_lr
                status = f"degrading ({self._bad_streak}/{self.down_patience}), LR stays at {current_lr:.2e}"

        self.lr_log.append(new_lr)
        print(
            f"  [SmartAdaptiveLR] Epoch {epoch + 1}: acc={val_acc:.4f} "
            f"loss={val_loss:.4f} | {reason} | {status}"
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
    axes[0, 0].set_title('Accuracy Trend (EfficientNetB0 Transfer Learning)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['loss'], label='Train Loss', color='royalblue')
    axes[0, 1].plot(history['val_loss'], label='Val Loss', color='tomato')
    add_boundary(axes[0, 1])
    axes[0, 1].set_title('Loss Trend (EfficientNetB0 Transfer Learning)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['precision'], label='Train Precision', color='royalblue')
    axes[1, 0].plot(history['val_precision'], label='Val Precision', color='tomato')
    add_boundary(axes[1, 0])
    axes[1, 0].set_title('Precision Trend (EfficientNetB0)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['recall'], label='Train Recall', color='royalblue')
    axes[1, 1].plot(history['val_recall'], label='Val Recall', color='tomato')
    add_boundary(axes[1, 1])
    axes[1, 1].set_title('Recall Trend (EfficientNetB0)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, 'training_history.png'))
    plt.show()


def merge_history(h1, h2):
    return {key: h1.history[key] + h2.history[key] for key in h1.history}


def plot_confusion_matrix_binary(model, val_gen):
    print("--- Generating confusion matrix ---")
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = (preds.flatten() > 0.5).astype(int)
    y_true = val_gen.classes

    idx_to_name = {v: k for k, v in val_gen.class_indices.items()}
    labels = [idx_to_name[0], idx_to_name[1]]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title('Confusion Matrix (EfficientNetB0 Transfer Learning)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, 'confusion_matrix.png'))
    plt.show()


def start_training():
    print(f"--- Starting training (EfficientNetB0 transfer learning), classes: {CLASSES} ---")
    print(f"--- train: {TRAIN_DIR}")
    print(f"--- val:   {VAL_DIR}")

    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=CLASSES,
        seed=SEED,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=CLASSES,
        seed=SEED,
        shuffle=False,
    )

    print(f"  Class indices: {train_gen.class_indices}")
    print(f"  Training samples: {train_gen.samples}, validation samples: {val_gen.samples}")

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            layers.Input(shape=(224, 224, 3)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        ],
        name='efficientnetb0_dirty_classifier',
    )
    model.summary()

    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]

    print(f"\n[Stage 1] Training classifier head for up to {EPOCHS_FROZEN} epochs...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
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

    print(
        f"\n[Stage 2] Unfreezing the last {UNFREEZE_LAYERS} EfficientNetB0 layers "
        f"for up to {EPOCHS_FINETUNE} epochs..."
    )
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=metrics,
    )
    lr_log_stage2 = []
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=make_callbacks(
            checkpoint_name=os.path.join(CURRENT_DIR, 'best_transfer_model.h5'),
            init_lr=1e-4,
            max_lr=5e-4,
            min_lr=1e-7,
            lr_log=lr_log_stage2,
        ),
    )

    print("\nTraining finished. Best model saved as best_transfer_model.h5")

    history = merge_history(history1, history2)
    stage1_end = len(history1.history['accuracy'])

    best_val_acc = max(history['val_accuracy'])
    best_val_prec = max(history['val_precision'])
    best_val_rec = max(history['val_recall'])
    best_val_f1 = 2 * best_val_prec * best_val_rec / (best_val_prec + best_val_rec + 1e-8)

    print("\nValidation summary")
    print(f"   Best Val Accuracy  : {best_val_acc:.4f}")
    print(f"   Best Val Precision : {best_val_prec:.4f}")
    print(f"   Best Val Recall    : {best_val_rec:.4f}")
    print(f"   Approx Val F1      : {best_val_f1:.4f}")

    plot_training_results(history, stage1_end)
    plot_confusion_matrix_binary(model, val_gen)


if __name__ == "__main__":
    start_training()
