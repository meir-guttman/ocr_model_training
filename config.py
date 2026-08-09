# ==========================================
# OCR Configuration File
# ==========================================
# כאן הבודק יעדכן את הנתיבים לפי המחשב שלו
# יש להקפיד על שימוש בלוכסן רגיל (/) בנתיבים

# נתיבי אימון (Train)
TRAIN_CSV = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TRAIN/train.csv'
TRAIN_IMG_DIR = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TRAIN/images'

# נתיבי אימות (Validation)
VAL_CSV = 'C:/dev/stam-app/our-ocr-engine/ocr-data/VAL/val.csv'
VAL_IMG_DIR = 'C:/dev/stam-app/our-ocr-engine/ocr-data/VAL/images'

# נתיבי בדיקה (Test)
TEST_CSV = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TEST/test.csv'
TEST_IMG_DIR = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TEST/images'

# נתיב לקובץ המודל (לשמירה ולטעינה)
MODEL_PATH = 'models/best_crnn_model.pth'

# נתיב לתמונה ספציפית עבור חיזוי (Predict)
PREDICT_IMAGE = 'C:/dev/stam-app/our-ocr-engine/ocr-data/TEST/images/sper_tura_Shorts_f4_004_line_007.jpg'

# פרמטרים נוספים שניתן לשנות בקלות
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
TARGET_HEIGHT = 64