"""
🌾 FASAL — Model Training Script (v2)
Uses Transfer Learning with EfficientNetB0 on the PlantVillage Dataset.
Also trains a Leaf Validation model to reject non-leaf images.

Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Usage:
    python train.py --data_dir ./data/PlantVillage --epochs 15 --batch_size 32
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report


def create_disease_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """
    Create an EfficientNetB0-based transfer learning model for disease classification.

    Why EfficientNetB0 over MobileNetV2:
    - Better accuracy-to-parameters ratio (compound scaling)
    - 77.1% top-1 ImageNet accuracy vs MobileNetV2's 71.8%
    - Only 5.3M parameters (efficient for deployment)

    Args:
        num_classes: Number of output classes
        input_shape: Input image dimensions

    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model initially
    base_model.trainable = False

    # Build classifier head with stronger regularization
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_leaf_validator(input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """
    Create a binary classifier: Leaf vs Not-Leaf.
    This acts as a gate before disease classification.
    Uses a lightweight custom CNN (no heavy base model needed for binary task).
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary: leaf or not
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_data_generators(data_dir: str, batch_size: int = 32, img_size: tuple = (224, 224)):
    """Create training and validation data generators with augmentation."""

    # IMPORTANT: EfficientNetB0 expects 0-255 pixel range and handles
    # its own normalization internally. Do NOT rescale to 0-1.
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    class_names = list(train_generator.class_indices.keys())
    return train_generator, val_generator, class_names


def generate_non_leaf_data(data_dir: str, num_samples: int = 2000, img_size: tuple = (224, 224)):
    """
    Generate synthetic non-leaf images for leaf validator training.
    Creates random noise, solid colors, gradient images, and scrambled patterns.
    """
    non_leaf_images = []
    labels = []

    for _ in range(num_samples // 4):
        # Random noise
        img = np.random.rand(img_size[0], img_size[1], 3).astype(np.float32)
        non_leaf_images.append(img)
        labels.append(0)  # 0 = not a leaf

        # Solid random color
        img = np.ones((img_size[0], img_size[1], 3), dtype=np.float32)
        img *= np.random.rand(3).astype(np.float32)
        non_leaf_images.append(img)
        labels.append(0)

        # Random gradient
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
        for c in range(3):
            gradient = np.linspace(np.random.rand(), np.random.rand(), img_size[0])
            img[:, :, c] = gradient[:, np.newaxis]
        non_leaf_images.append(img)
        labels.append(0)

        # Skin-tone colored blocks (to reject human faces/hands)
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
        # Skin tone range in RGB
        img[:, :, 0] = np.random.uniform(0.7, 1.0)  # High red
        img[:, :, 1] = np.random.uniform(0.4, 0.7)  # Medium green
        img[:, :, 2] = np.random.uniform(0.3, 0.6)  # Lower blue
        # Add noise
        img += np.random.normal(0, 0.05, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)
        non_leaf_images.append(img)
        labels.append(0)

    # Load some real leaf images as positive examples
    leaf_images = []
    count = 0
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            if count >= num_samples:
                break
            try:
                img_path = os.path.join(class_path, img_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                leaf_images.append(img)
                labels.append(1)  # 1 = leaf
                count += 1
            except Exception:
                continue
        if count >= num_samples:
            break

    all_images = np.array(non_leaf_images + leaf_images, dtype=np.float32)
    all_labels = np.array(labels[:len(non_leaf_images)] + labels[len(non_leaf_images):], dtype=np.float32)

    # Shuffle
    indices = np.random.permutation(len(all_images))
    return all_images[indices], all_labels[indices]


def fine_tune_model(model: tf.keras.Model, learning_rate: float = 1e-5, fine_tune_at: int = 200):
    """Unfreeze top layers of base model for fine-tuning."""
    base_model = model.layers[0]
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_training_history(history, save_path: str = 'training_history.png'):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train FASAL Disease Detection Models')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to PlantVillage dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs (default: 15)')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Fine-tuning epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--skip_validator', action='store_true', help='Skip leaf validator training')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  🌾 FASAL — AI-Driven Crop Disease Detector")
    print("  Model Training Pipeline v2 (EfficientNetB0)")
    print("=" * 60)

    # ═══════════════════════════════════════════════
    # PART 1: Train Leaf Validator
    # ═══════════════════════════════════════════════
    if not args.skip_validator:
        print("\n" + "─" * 60)
        print("  PART 1: Training Leaf Validator (Leaf vs Not-Leaf)")
        print("─" * 60)

        print("\n📂 Generating training data for leaf validator...")
        X, y = generate_non_leaf_data(args.data_dir, num_samples=2000)
        print(f"   Total samples: {len(X)} (Leaves: {int(y.sum())}, Non-leaves: {int(len(y) - y.sum())})")

        # Split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print("\n🏗️  Building Leaf Validator CNN...")
        validator = create_leaf_validator()
        validator.summary()

        print("\n🚀 Training Leaf Validator...")
        val_history = validator.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
            ],
            verbose=1
        )

        validator_path = os.path.join(args.output_dir, 'leaf_validator.keras')
        validator.save(validator_path)
        val_acc = validator.evaluate(X_val, y_val, verbose=0)[1]
        print(f"\n✅ Leaf Validator saved! Validation Accuracy: {val_acc:.4f}")

        # Clean up memory
        del X, y, X_train, X_val, y_train, y_val
    else:
        print("\n⏭️  Skipping leaf validator training (--skip_validator flag)")

    # ═══════════════════════════════════════════════
    # PART 2: Train Disease Classifier (EfficientNetB0)
    # ═══════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  PART 2: Training Disease Classifier (EfficientNetB0)")
    print("─" * 60)

    print("\n📂 Loading dataset...")
    train_gen, val_gen, class_names = create_data_generators(args.data_dir, args.batch_size)
    num_classes = len(class_names)
    print(f"   Found {num_classes} classes")
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")

    # Save class names
    class_names_path = os.path.join(args.output_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)

    # Save model metadata
    metadata = {
        "model_name": "FASAL Disease Classifier v2",
        "base_model": "EfficientNetB0",
        "num_classes": num_classes,
        "input_size": 224,
        "training_samples": train_gen.samples,
        "validation_samples": val_gen.samples,
        "has_leaf_validator": not args.skip_validator
    }
    with open(os.path.join(args.output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n🏗️  Building EfficientNetB0 model...")
    model = create_disease_model(num_classes)
    model.summary()

    model_callbacks = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]

    # Phase 1: Feature Extraction
    print("\n🚀 Phase 1: Feature Extraction...")
    history1 = model.fit(
        train_gen, epochs=args.epochs,
        validation_data=val_gen,
        callbacks=model_callbacks, verbose=1
    )

    # Phase 2: Fine-Tuning
    print("\n🔧 Phase 2: Fine-Tuning top layers...")
    model = fine_tune_model(model)
    history2 = model.fit(
        train_gen, epochs=args.fine_tune_epochs,
        validation_data=val_gen,
        callbacks=model_callbacks, verbose=1
    )

    # Merge histories
    combined = type(history1)()
    combined.history = {}
    for key in history1.history:
        combined.history[key] = history1.history[key] + history2.history[key]

    # Evaluate
    print("\n📊 Evaluating model...")
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes

    report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    with open(os.path.join(args.output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Save final model
    final_path = os.path.join(args.output_dir, 'plant_disease_model.keras')
    model.save(final_path)
    print(f"\n✅ Disease model saved to {final_path}")

    # TFLite conversion
    print("\n📱 Converting to TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = os.path.join(args.output_dir, 'plant_disease_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"   TFLite model saved to {tflite_path}")
    except Exception as e:
        print(f"   TFLite conversion skipped: {e}")

    # Plot training history
    plot_training_history(combined, os.path.join(args.output_dir, 'training_history.png'))

    # Final summary
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print("\n" + "=" * 60)
    print(f"  🎉 FASAL Training Complete!")
    print(f"  📈 Final Validation Accuracy: {val_acc:.4f}")
    print(f"  📉 Final Validation Loss:     {val_loss:.4f}")
    print(f"  🧠 Disease Model:  {final_path}")
    if not args.skip_validator:
        print(f"  🛡️  Leaf Validator: {os.path.join(args.output_dir, 'leaf_validator.keras')}")
    print(f"  📁 All outputs:    {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
