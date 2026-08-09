import torch
import cv2
import pandas as pd
import os
import Levenshtein  # ספרייה לחישוב מרחק בין מחרוזות (חובה להתקין)

# ייבוא מהקבצים שלנו
from crnn_model import CRNN
from ocr_dataset import HEBREW_VOCAB
from predict import preprocess_image, decode_predictions
from config import *


VOCAB_SIZE = len(HEBREW_VOCAB)


def calculate_cer(predicted_text, true_text):
    """
    חישוב CER - Character Error Rate
    כמה פעולות (הוספה/מחיקה/החלפה) נדרשות כדי להפוך את הניחוש לטקסט האמיתי,
    לחלק במספר התווים בטקסט האמיתי.
    """
    if len(true_text) == 0:
        return 1.0  # למנוע חלוקה באפס

    distance = Levenshtein.distance(predicted_text, true_text)
    cer = distance / len(true_text)
    return cer


def evaluate_model():
    print("Loading Test Dataset...")
    if not os.path.exists(TEST_CSV):
        print(f"Error: Could not find {TEST_CSV}")
        return

    # טעינת הנתונים
    df = pd.read_csv(TEST_CSV)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on: {device}")

    # טעינת המודל
    model = CRNN(vocab_size=VOCAB_SIZE, hidden_size=256).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    total_characters = 0
    total_errors = 0

    print("-" * 50)
    print("Starting Evaluation...")
    print("-" * 50)

    # מעבר על כל התמונות בסט ה-TEST
    for index, row in df.iterrows():
        img_name = str(row.iloc[0])  # עמודה ראשונה: שם קובץ
        true_text = str(row.iloc[1])  # עמודה שנייה: טקסט אמיתי

        img_path = os.path.join(TEST_IMG_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue

        # הכנת התמונה וריצת המודל
        img_tensor = preprocess_image(img_path)
        if img_tensor is None:
            continue

        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        predicted_text = decode_predictions(outputs)

        # חישוב שגיאות לתמונה הספציפית
        distance = Levenshtein.distance(predicted_text, true_text)
        total_errors += distance
        total_characters += len(true_text)

        # הדפסת מקרים שבהם המודל טעה (כדי שנוכל ללמוד מהם)
        if distance > 0:
            print(f"File: {img_name}")
            print(f"True : {true_text}")
            print(f"Pred : {predicted_text}")
            print(f"Errors: {distance} characters")
            print("-")

    # חישוב התוצאה הסופית
    if total_characters > 0:
        final_cer = total_errors / total_characters
        accuracy = (1.0 - final_cer) * 100
        print("=" * 50)
        print("FINAL TEST RESULTS:")
        print(f"Total Images Evaluated: {len(df)}")
        print(f"Total Characters: {total_characters}")
        print(f"Total Errors: {total_errors}")
        print(f"Character Error Rate (CER): {final_cer:.4f}")
        print(f"Character Accuracy: {accuracy:.2f}%")
        print("=" * 50)
    else:
        print("No characters evaluated.")


if __name__ == '__main__':
    evaluate_model()
