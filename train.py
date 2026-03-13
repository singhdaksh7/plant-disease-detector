"""
Plant Disease Detection - Model Training Script
Uses Transfer Learning with MobileNetV2 on the PlantVillage Dataset.

Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
         (or use tensorflow_datasets: 'plant_village')

Usage:
    python train.py --data_dir ./data/PlantVillage --epochs 20 --batch_size 32
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


def create_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """
    Create a MobileNetV2-based transfer learning model.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image dimensions
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers initially
    base_model.trainable = False
    
    # Build the full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_data_generators(data_dir: str, batch_size: int = 32, img_size: tuple = (224, 224)):
    """
    Create training and validation data generators with augmentation.
    
    Args:
        data_dir: Path to the PlantVillage dataset directory
        batch_size: Batch size for training
        img_size: Target image size
    
    Returns:
        train_generator, val_generator, class_names
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
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


def fine_tune_model(model: tf.keras.Model, learning_rate: float = 1e-5, fine_tune_at: int = 100):
    """
    Unfreeze top layers of base model for fine-tuning.
    
    Args:
        model: The pre-trained model
        learning_rate: Learning rate for fine-tuning (should be very small)
        fine_tune_at: Layer number to start fine-tuning from
    """
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers before `fine_tune_at`
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, save_path: str = 'training_history.png'):
    """Plot and save training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to PlantVillage dataset directory')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save the trained model')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  🌿 Plant Disease Detection - Model Training")
    print("=" * 60)
    
    # ── Step 1: Load Data ──
    print("\n📂 Loading dataset...")
    train_gen, val_gen, class_names = create_data_generators(
        args.data_dir, args.batch_size
    )
    num_classes = len(class_names)
    print(f"   Found {num_classes} classes")
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    
    # Save class names
    class_names_path = os.path.join(args.output_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"   Class names saved to {class_names_path}")
    
    # ── Step 2: Create & Train Model (Feature Extraction) ──
    print("\n🏗️  Building MobileNetV2 model...")
    model = create_model(num_classes)
    model.summary()
    
    # Callbacks
    model_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n🚀 Phase 1: Feature Extraction Training...")
    history1 = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=model_callbacks,
        verbose=1
    )
    
    # ── Step 3: Fine-Tuning ──
    print("\n🔧 Phase 2: Fine-Tuning top layers...")
    model = fine_tune_model(model)
    
    history2 = model.fit(
        train_gen,
        epochs=args.fine_tune_epochs,
        validation_data=val_gen,
        callbacks=model_callbacks,
        verbose=1
    )
    
    # Merge histories
    combined_history = type(history1)()
    combined_history.history = {}
    for key in history1.history:
        combined_history.history[key] = history1.history[key] + history2.history[key]
    
    # ── Step 4: Evaluate ──
    print("\n📊 Evaluating model...")
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    # Classification report
    report = classification_report(
        true_classes, predicted_classes,
        target_names=class_names,
        output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Save report
    report_path = os.path.join(args.output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # ── Step 5: Save Final Model ──
    final_model_path = os.path.join(args.output_dir, 'plant_disease_model.keras')
    model.save(final_model_path)
    print(f"\n✅ Model saved to {final_model_path}")
    
    # Save as TFLite for mobile deployment (optional)
    print("\n📱 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(args.output_dir, 'plant_disease_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"   TFLite model saved to {tflite_path}")
    
    # Plot training history
    plot_training_history(
        combined_history,
        os.path.join(args.output_dir, 'training_history.png')
    )
    
    # Final summary
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print("\n" + "=" * 60)
    print(f"  🎉 Training Complete!")
    print(f"  📈 Final Validation Accuracy: {val_acc:.4f}")
    print(f"  📉 Final Validation Loss:     {val_loss:.4f}")
    print(f"  📁 All outputs saved to:      {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
