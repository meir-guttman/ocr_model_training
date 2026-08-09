import torch
import cv2
import numpy as np
import os

# ייבוא הארכיטקטורה והמילון שלנו
from crnn_model import CRNN
from ocr_dataset import HEBREW_VOCAB, IDX_TO_CHAR
from config import *

VOCAB_SIZE = len(HEBREW_VOCAB)


def preprocess_image(image_path):
    """
    טעינת תמונה חדשה מהדיסק והכנתה לפורמט שהמודל מכיר.
    """
    # קריאת התמונה בגווני אפור
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # 1. היפוך אופקי (חובה לעברית - בדיוק כמו שעשינו באימון)
    image = cv2.flip(image, 1)
    
    # 2. שינוי גודל לגובה 64 פיקסלים תוך שמירה על פרופורציות (Aspect Ratio)
    h, w = image.shape
    aspect_ratio = w / h
    new_w = int(TARGET_HEIGHT * aspect_ratio)
    image = cv2.resize(image, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # 3. המרה ל-Tensor ונורמליזציה לערכים בין 0 ל-1
    # נוסיף את מימד האצווה (Batch=1) ומימד הערוץ (Channels=1)
    # התוצאה: [1, 1, 64, Width]
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    return image_tensor

def decode_predictions(predictions):
    """
    ממיר את ההסתברויות שהמודל פולט חזרה לטקסט רגיל (Greedy Decoding).
    ב-CTC אנו חותכים כפילויות רצופות ומתעלמים מתווים ריקים (Blank).
    """
    # predictions מגיע בצורה: [Sequence_Length, Batch_Size, Vocab_Size]
    # מכיוון שאנו בודקים תמונה אחת, ה-Batch הוא 1.
    # ניקח את האינדקס עם ההסתברות הגבוהה ביותר (argmax) לכל צעד ברצף
    preds = predictions.argmax(dim=2).squeeze(1) # התוצאה: [Sequence_Length]
    
    decoded_text = []
    prev_char_idx = -1
    
    for p in preds:
        idx = p.item()
        # תנאי 1: האינדקס שונה מ-0 (0 מוגדר כ-Blank token אצלנו)
        # תנאי 2: האינדקס שונה מהאינדקס הקודם (מונע כפילות כמו "בברראששייתת")
        if idx != 0 and idx != prev_char_idx:
            # משיכת התו האמיתי מהמילון
            decoded_text.append(IDX_TO_CHAR[idx])
        
        # שומרים את התו הנוכחי כדי להשוות בסיבוב הבא
        prev_char_idx = idx
        
    # חיבור כל התווים למחרוזת אחת
    return "".join(decoded_text)

def predict(image_path):
    """
    פונקציה מלאה לזיהוי טקסט מתמונה אחת.
    """
    # בדיקה האם קובץ המודל קיים בכלל
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}.")
        print("Please wait for the training to finish and save the best model.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running prediction on: {device}")
    
    # 1. יצירת רשת ה-CRNN הריקה (חובה להשתמש באותם ממדים כמו באימון)
    model = CRNN(vocab_size=VOCAB_SIZE, hidden_size=256).to(device)
    
    # 2. טעינת המשקולות שהרשת למדה לתוך המודל
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # 3. מעבר למצב חיזוי (חשוב מאוד כי זה מכבה שכבות שרלוונטיות רק לאימון כמו Dropout)
    model.eval() 
    
    # 4. הכנת התמונה
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return
        
    img_tensor = img_tensor.to(device)
    
    # 5. ריצת המודל
    # עוטפים ב-no_grad כי אנחנו לא צריכים לחשב שגיאות/גרדיאנטים (חוסך המון זיכרון וזמן)
    with torch.no_grad():
        outputs = model(img_tensor)
        
    # 6. פענוח והדפסה
    result_text = decode_predictions(outputs)
    
    print("-" * 30)
    print(f"File: {image_path}")
    print(f"Predicted Text: {result_text}")
    print("-" * 30)

if __name__ == '__main__':
    # דוגמה להרצה. שנה את הנתיב לתמונה מתוך סט האימות או תמונה חדשה שגזרת:
    # נניח: test_image = 'VAL/images/document_1_line_002.jpg'

    
    if os.path.exists(PREDICT_IMAGE):
        predict(PREDICT_IMAGE)
    else:
        print(f"Please update the path '{PREDICT_IMAGE}' to an actual image from your dataset.")