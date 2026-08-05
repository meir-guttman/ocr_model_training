import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# מילון התווים שלנו. 
# אינדקס 0 שמור ל-CTC Blank Token (חובה באלגוריתם CTC).
# אינדקס 1 שמור לרווח בין מילים.
HEBREW_VOCAB = {
    '<blank>': 0,
    ' ': 1,
    'א': 2, 'ב': 3, 'ג': 4, 'ד': 5, 'ה': 6, 'ו': 7, 'ז': 8, 'ח': 9, 'ט': 10,
    'י': 11, 'כ': 12, 'ך': 13, 'ל': 14, 'מ': 15, 'ם': 16, 'נ': 17, 'ן': 18,
    'ס': 19, 'ע': 20, 'פ': 21, 'ף': 22, 'צ': 23, 'ץ': 24, 'ק': 25, 'ר': 26, 
    'ש': 27, 'ת': 28
}

# יצירת מילון הפוך (ממספר לתו) כדי שנוכל לפענח את התשובות של המודל אחר כך
IDX_TO_CHAR = {v: k for k, v in HEBREW_VOCAB.items()}

class StamOCRDataset(Dataset):
    def __init__(self, csv_file, img_dir, target_height=64):
        """
        csv_file: נתיב לקובץ ה-CSV המכיל את שמות הקבצים והטקסטים (למשל train.csv)
        img_dir: נתיב לספרייה שמכילה את התמונות בפועל
        target_height: הגובה שאליו ננרמל את כל התמונות (סטנדרט של 64 או 32)
        """
        self.data_df = pd.read_csv(csv_file)
        # ודא שהעמודות ב-CSV אכן נקראות 'file_name' ו-'text'
        self.img_names = self.data_df.iloc[:, 0].values
        self.texts = self.data_df.iloc[:, 1].values
        self.img_dir = img_dir
        self.target_height = target_height

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # 1. קריאת התמונה
        img_name = str(self.img_names[idx])
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # 2. הפיכת התמונה (Flipping) - קריטי לטיפול בעברית (RTL)
        # 1 מסמן היפוך אופקי. כך המודל שסורק משמאל לימין יפגוש את תחילת המילה קודם.
        image = cv2.flip(image, 1)

        # 3. נורמליזציה של גובה (Resize) שמירה על יחס רוחב-גובה (Aspect Ratio)
        h, w = image.shape
        aspect_ratio = w / h
        new_w = int(self.target_height * aspect_ratio)
        
        # מבצעים את שינוי הגודל
        image = cv2.resize(image, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

        # 4. המרת התמונה ל-Tensor ונורמליזציה (ערכים בין 0 ל-1)
        # מוסיפים מימד נוסף עבור ה"ערוץ" (Channel), כי PyTorch מצפה ל-[Channels, Height, Width]
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        # 5. קידוד הטקסט למספרים
        text = str(self.texts[idx])
        text_encoded = [HEBREW_VOCAB[char] for char in text if char in HEBREW_VOCAB]
        text_tensor = torch.tensor(text_encoded, dtype=torch.long)

        # מחזירים את התמונה והטקסט, וכן את הרוחב המקורי של התמונה המנורמלת (יעזור לנו ב-Collate)
        return image_tensor, text_tensor, new_w

def ocr_collate_fn(batch):
    """
    פונקציה זו רצה על כל Batch (למשל 16 תמונות) לפני שהן נכנסות ל-GPU.
    מכיוון שהשורות שלנו באורכים שונים, אנחנו מוצאים את הרוחב המקסימלי באותו Batch,
    ומרפדים (Padding) את שאר התמונות כדי שיהיו באותו רוחב.
    """
    images, texts, widths = zip(*batch)
    
    # מציאת הרוחב המקסימלי ב-Batch הנוכחי
    max_width = max(widths)
    target_height = images[0].shape[1]
    
    # יצירת טנזור ריק בגודל [Batch_Size, 1, Target_Height, Max_Width] מרופד בלבן (או שחור, תלוי בנתונים)
    # נניח שהרקע שלך הוא לבן (1.0), אז נמלא באחדות:
    padded_images = torch.ones((len(images), 1, target_height, max_width), dtype=torch.float32)
    
    # הכנסת התמונות לתוך הטנזור המרופד
    for i, img in enumerate(images):
        w = img.shape[2]
        padded_images[i, :, :, :w] = img

    # CTC Loss של PyTorch מצריך לדעת מה האורך האמיתי של כל תמונה (ללא הריפוד)
    # ומה האורך האמיתי של כל טקסט.
    input_lengths = torch.tensor(widths, dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    
    # שרשור כל מספרי הטקסט לטנזור ארוך אחד (זו הדרישה של CTC Loss ב-PyTorch)
    targets = torch.cat(texts)

    return padded_images, targets, input_lengths, target_lengths

if __name__ == '__main__':
    # דוגמה לאיך לקרוא לפונקציות האלו (תצטרך לשנות את הנתיבים בהתאם למחשב שלך)
    # שים לב שזה רק לבדיקה ראשונית
    
    TRAIN_CSV = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TRAIN/train.csv'
    TRAIN_IMG_DIR = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TRAIN/images'
    
    # בדיקה שקובץ ה-CSV קיים כדי למנוע קריסה
    if os.path.exists(TRAIN_CSV):
        dataset = StamOCRDataset(csv_file=TRAIN_CSV, img_dir=TRAIN_IMG_DIR, target_height=64)
        
        # יצירת DataLoader - אחראי על טעינה מקבילית וחיתוך לאצוות (Batches)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=ocr_collate_fn)
        
        # הוצאת Batch ראשון לבדיקה
        images, targets, input_lengths, target_lengths = next(iter(dataloader))
        
        print("Dataset initialized successfully!")
        print(f"Batch images shape: {images.shape} -> [Batch, Channels, Height, Max_Width]")
        print(f"Targets tensor: {targets}")
        print(f"Input lengths (original widths): {input_lengths}")
        print(f"Target lengths (number of chars per line): {target_lengths}")
    else:
        print(f"Test skipped. Please update the path '{TRAIN_CSV}' to match your directory structure.")