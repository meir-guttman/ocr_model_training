import sys
from config import PREDICT_IMAGE

# ייבוא הפונקציות המרכזיות מהסקריפטים שכבר בנינו
from train_ocr import train_model
from evaluate_test import evaluate_model
from predict import predict


def main_menu():
    print("========================================")
    print("      OCR Training & Testing Tool       ")
    print("========================================")
    print("1. Train Model (Continue or Start Fresh)")
    print("2. Evaluate Model on Test Set (CER)")
    print("3. Predict a Single Image")
    print("4. Exit")
    print("========================================")

    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        print("\nStarting Training Process...")
        train_model()
    elif choice == '2':
        print("\nStarting Evaluation Process...")
        evaluate_model()
    elif choice == '3':
        print(f"\nRunning Prediction on: {PREDICT_IMAGE}")
        predict(PREDICT_IMAGE)
    elif choice == '4':
        print("\nExiting...")
        sys.exit(0)
    else:
        print("\nInvalid choice. Please try again.")


if __name__ == '__main__':
    while True:
        main_menu()
        print("\n")