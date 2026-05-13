import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

DATASET_PATH = "dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, 'dataset_final', 'train')
TEST_DIR = os.path.join(DATASET_PATH, 'dataset_final', 'test')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 40

NUM_CLASSES = 8
CLASS_NAMES = [
    '0_Sneakers',
    '1_Flat_Shoes',
    '2_Slip_Ons',
    '3_Sandals',
    '4_High_Heels',
    '5_Boots',
    '6_Rubber_Slippers',
    '7_Cotton_Slippers',
]

print("Configuring data generators...")

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True,
    seed=42,
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False,
)

print(f"Training samples: {train_generator.samples}, test samples: {test_generator.samples}")
print(f"Class mapping: {train_generator.class_indices}")
print("\nBuilding MobileNetV2 transfer learning model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

lr_log_stage1 = []
lr_log_stage2 = []


class SmartAdaptiveLR(tf.keras.callbacks.Callback):
    def __init__(self, init_lr, max_lr, min_lr, up_factor=1.2, down_factor=0.5, patience=2, lr_log=None):
        super().__init__()
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.up_factor = up_factor
        self.down_factor = down_factor
        self.patience = patience
        self.lr_log = lr_log if lr_log is not None else []
        self._best_acc = -float('inf')
        self._best_loss = float('inf')
        self._good_streak = 0

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.init_lr)
        self._best_acc = -float('inf')
        self._best_loss = float('inf')
        self._good_streak = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy', None)
        val_loss = logs.get('val_loss', None)

        if val_acc is None or val_loss is None:
            return

        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        acc_improved = val_acc > self._best_acc + 1e-4
        loss_improved = val_loss < self._best_loss - 1e-4
        acc_worse = val_acc < self._best_acc - 1e-3
        loss_worse = val_loss > self._best_loss + 1e-3

        if acc_improved and loss_improved:
            self._best_acc = val_acc
            self._best_loss = val_loss
            self._good_streak += 1

            if self._good_streak >= self.patience:
                new_lr = min(current_lr * self.up_factor, self.max_lr)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                self._good_streak = 0
                status = f"consistent improvement, LR raised from {current_lr:.2e} to {new_lr:.2e}"
            else:
                new_lr = current_lr
                status = f"improving ({self._good_streak}/{self.patience}), LR stays at {current_lr:.2e}"
        elif acc_worse or loss_worse:
            new_lr = max(current_lr * self.down_factor, self.min_lr)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            self._good_streak = 0
            status = f"performance dropped, LR lowered from {current_lr:.2e} to {new_lr:.2e}"
        else:
            self._good_streak = 0
            new_lr = current_lr
            status = f"plateau detected, LR stays at {current_lr:.2e}"

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
            lr_log=lr_log,
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_name,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]


print(f"\n[Stage 1] Training classifier head for {EPOCHS_FROZEN} epochs...")

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_FROZEN,
    validation_data=test_generator,
    callbacks=make_callbacks(
        checkpoint_name='best_stage1.h5',
        init_lr=1e-3,
        max_lr=5e-3,
        min_lr=1e-6,
        lr_log=lr_log_stage1,
    ),
)

print("\n[Stage 2] Unfreezing the last 30 MobileNetV2 layers for fine-tuning...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_FINETUNE,
    validation_data=test_generator,
    callbacks=make_callbacks(
        checkpoint_name='best_shoes_classifier.h5',
        init_lr=1e-4,
        max_lr=5e-4,
        min_lr=1e-7,
        lr_log=lr_log_stage2,
    ),
)


def merge_history(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged


history = merge_history(history1, history2)
stage1_end = len(history1.history['accuracy'])

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(history['accuracy'], label='Train Acc', color='royalblue')
plt.plot(history['val_accuracy'], label='Val Acc', color='tomato')
plt.axvline(x=stage1_end - 1, color='gray', linestyle='--', label='Fine-tune Start')
plt.title('Accuracy Trend')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history['loss'], label='Train Loss', color='royalblue')
plt.plot(history['val_loss'], label='Val Loss', color='tomato')
plt.axvline(x=stage1_end - 1, color='gray', linestyle='--', label='Fine-tune Start')
plt.title('Loss Trend')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
all_lr = lr_log_stage1 + lr_log_stage2
epochs_axis = list(range(1, len(all_lr) + 1))
plt.plot(epochs_axis[:len(lr_log_stage1)], lr_log_stage1, color='royalblue', label='Stage 1')
plt.plot(epochs_axis[len(lr_log_stage1):], lr_log_stage2, color='darkorange', label='Stage 2')
plt.axvline(x=stage1_end, color='gray', linestyle='--', label='Fine-tune Start')
plt.yscale('log')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate (log scale)')
plt.legend()
plt.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Training curves saved to training_history.png")

print("\nGenerating classification report...")

model.load_weights('best_shoes_classifier.h5')
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(NUM_CLASSES)
short_names = [name.split('_', 1)[1] for name in CLASS_NAMES]
plt.xticks(tick_marks, short_names, rotation=45, ha='right')
plt.yticks(tick_marks, short_names)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], 'd'),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix.png")

model.save('final_shoe_classifier.h5')
print("\nTraining completed.")
print("   Best model (validation): best_shoes_classifier.h5")
print("   Final model (last epoch): final_shoe_classifier.h5")
print("   Training curves: training_history.png")
print("   Confusion matrix: confusion_matrix.png")
