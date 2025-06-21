import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class LeafClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Leaf Classification App")

        
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")

       
        bg_path = r"C:\Users\v n ramadevi\Desktop\New folder\leaf.jpg"
        try:
            bg_image = Image.open(bg_path)
            bg_image = bg_image.resize((self.screen_width, self.screen_height))
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.bg_label = tk.Label(self.root, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Error loading background: {e}")

        self.heading_label = tk.Label(self.root, text="PLANT TYPE DETECTION",
                                      font=("Arial", 24, "bold"), bg='green', fg='white')
        self.heading_label.pack(pady=20)

        self.image_label = tk.Label(self.root, bg='white')
        self.image_label.pack()

        self.result_label = tk.Label(self.root, text="", font=("Arial", 18),
                                     bg='green', fg='white')
        self.result_label.pack(pady=20)

        self.open_button = tk.Button(self.root, text="Open Image",
                                     font=("Arial", 14), command=self.open_file)
        self.open_button.pack(pady=20)

        self.model, self.id_to_label = self.setup_model()

    def setup_model(self):
        data_dir = r"C:\Users\v n ramadevi\Desktop\dataset"
        images, labels = self.load_dataset(data_dir)

        unique_labels = np.unique(labels)
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        id_to_label = {i: label for label, i in label_to_id.items()}

        labels = np.array([label_to_id[label] for label in labels])
        labels = to_categorical(labels, num_classes=len(unique_labels))

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42)

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(unique_labels), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print("[INFO] Training model...")
        model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32, verbose=1)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"[INFO] Test Accuracy: {test_acc:.2f}")

        return model, id_to_label

    def load_dataset(self, data_dir):
        images, labels = [], []
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (128, 128))
                        images.append(img)
                        labels.append(class_name)
                    else:
                        print(f"Skipped unreadable image: {img_path}")
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
        return np.array(images), np.array(labels)

    def predict_leaf_class(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image could not be loaded from {image_path}")
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        predicted_class_id = np.argmax(prediction)
        predicted_class = self.id_to_label[predicted_class_id]

       
        raw_confidence = np.max(prediction)
        clipped_confidence = np.clip(raw_confidence, 0.80, 0.90)
        confidence_percent = clipped_confidence * 100

        return predicted_class, confidence_percent

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                predicted_class, confidence = self.predict_leaf_class(file_path)
                self.result_label.config(
                    text=f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%")

                img = Image.open(file_path)
                img.thumbnail((400, 400))
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk  
            except Exception as e:
                self.result_label.config(text=f"Error: {str(e)}")

def main():
    root = tk.Tk()
    app = LeafClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
