import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# ייבוא מהקבצים האחרים שכתבנו
from crnn_model import CRNN
from ocr_dataset import StamOCRDataset, ocr_collate_fn, HEBREW_VOCAB

# הגדרת נתיבים (תעדכן אם השמות אצלך קצת שונים)
TRAIN_CSV = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TRAIN/train.csv'
TRAIN_IMG_DIR = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TRAIN/images'

VAL_CSV = 'C:/dev/stam-app/our-ocr-engine/ocr-data/VAL/val.csv'
VAL_IMG_DIR = 'C:/dev/stam-app/our-ocr-engine/ocr-data/VAL/images'

# פרמטרים של האימון (Hyperparameters)
BATCH_SIZE = 16          # כמה תמונות המודל רואה בכל פעם
NUM_EPOCHS = 50          # כמה פעמים המודל יעבור על כל ה-831 תמונות
LEARNING_RATE = 0.001    # קצב הלמידה
TARGET_HEIGHT = 64
VOCAB_SIZE = len(HEBREW_VOCAB) # 29 (27 אותיות + רווח + Blank)

def train_model():
    # בחירת כרטיס מסך (GPU) אם קיים, אחרת שימוש במעבד (CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Starting training on device: {device}')

    # 1. טעינת הדאטה-סט לאימון
    train_dataset = StamOCRDataset(TRAIN_CSV, TRAIN_IMG_DIR, target_height=TARGET_HEIGHT)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ocr_collate_fn)

    # טעינת הדאטה-סט לאימות (Validation)
    val_dataset = StamOCRDataset(VAL_CSV, VAL_IMG_DIR, target_height=TARGET_HEIGHT)
    # שים לב שב-Validation לא עושים shuffle (ערבוב), זה לא נחוץ
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ocr_collate_fn)

    print(f"Loaded {len(train_dataset)} train images and {len(val_dataset)} validation images.")

    # 2. יצירת המודל
    model = CRNN(vocab_size=VOCAB_SIZE, hidden_size=256).to(device)

    # The path to your saved model weights
    saved_model_path = "models/best_crnn_model.pth"

    if os.path.exists(saved_model_path):
        # File exists: Resume training
        print("Found existing model. Resuming training from previous state...")
        model.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu')))
        print("Successfully loaded previous model weights. Resuming training...")

    else:
        # File doesn't exist: Start fresh
        print("No previous model found. Starting training from scratch...")


    # 3. הגדרת פונקציית שגיאה ואופטימיזציה
    # blank=0 אומר שאינדקס 0 במילון מיועד לתו ה-Blank של אלגוריתם ה-CTC
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- תוספת קריטית 1: Scheduler ---
    # ה-Scheduler יקטין את קצב הלמידה בחצי אם ה-Val Loss נתקע (פלטו) במשך 3 Epochs
    # זה מה שעוזר למודל לצאת ממלכודת ה-Blank ולהתחיל להתכוונן עדין יותר!
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_val_loss = float('inf')

    # 4. לולאת האימון המרכזית
    for epoch in range(NUM_EPOCHS):
        model.train() # מעבר למצב אימון
        running_loss = 0.0

        for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(train_loader):
            # העברת הנתונים ל-GPU (או CPU)
            images = images.to(device)
            targets = targets.to(device)

            # איפוס גרדיאנטים
            optimizer.zero_grad()

            # העברה קדימה (Forward Pass)
            outputs = model(images)

            # חישוב אורך הפלט של הרשת (הרוחב התכווץ בגלל ה-MaxPool)
            # מכיוון שב-CNN שלנו יש 2 שכבות של Pool(2,2) ושכבת הפחתה 1 נוספת, רוחב התמונה מחולק ב-4.
            # הנוסחה המדויקת לארכיטקטורה הספציפית שבנינו:
            output_seq_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)

            # חישוב השגיאה (Loss)
            loss = criterion(outputs, targets, output_seq_lengths, target_lengths)

            # העברה אחורה (Backward Pass) ועדכון משקולות
            loss.backward()
            
            # --- תוספת קריטית 2: Gradient Clipping ---
            # מונע מהגרדיאנטים הנגזרים (המתמטיקה של הלמידה) "להתפוצץ" למספרים ענקיים
            # תופעה זו נפוצה מאוד ב-RNN ו-CTC וגורמת למודל להיתקע
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # עדכון משקולות
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)

        # --- שלב האימות (Validation) בסוף כל Epoch ---
        model.eval() # מעבר למצב חיזוי (מכבה שכבות כמו Dropout אם היו)
        val_loss = 0.0

        # לא מחשבים גרדיאנטים ב-Validation (חוסך זיכרון ומאיץ את הריצה)
        with torch.no_grad():
            for images, targets, input_lengths, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                output_seq_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)
                
                loss = criterion(outputs, targets, output_seq_lengths, target_lengths)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        print(f"===> Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # --- הפעלת ה-Scheduler ---
        # ה-Scheduler בודק את ה-val_loss ומחליט האם להקטין את קצב הלמידה
        scheduler.step(val_loss)

        # שמירת המודל הטוב ביותר
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # יוצר תיקייה למודלים אם לא קיימת
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_crnn_model.pth")
            print(f"     *** New Best Model Saved! ***")

if __name__ == '__main__':
    train_model()