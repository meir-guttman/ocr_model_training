import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()
        
        # 1. שכבת ה-CNN (מחלץ מאפיינים מהתמונה)
        # הקלט שלנו: תמונה בגובה 64 פיקסלים וערוץ צבע 1 (שחור-לבן)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # מקטין גובה ל-32 ורוחב לחצי
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # מקטין גובה ל-16 ורוחב לרבע (W/4)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # מכאן אנחנו מקטינים רק את הגובה ולא את הרוחב (כדי לשמור על רזולוציית רצף הטקסט)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # מקטין גובה ל-8
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # מקטין גובה ל-4
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), # מקטין גובה ל-3
        )

        # 2. שכבת ה-RNN (Bi-LSTM) שמבינה את הרצף
        # 512 (ערוצים) כפול 3 (הגובה שנשאר מה-CNN) = 1536 מאפיינים לכל עמודה ברוחב
        self.rnn = nn.LSTM(512 * 3, hidden_size, bidirectional=True, num_layers=2)
        
        # 3. שכבת הפלט - חיזוי לכל תו במילון
        # hidden_size * 2 בגלל שהשתמשנו ב-bidirectional=True (דו-כיווני)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # x.shape: [Batch_Size, 1, 64, Width]
        
        # העברה ב-CNN
        conv_out = self.cnn(x) 
        # conv_out.shape: [Batch_Size, 512, 3, Width / 4 - 1]
        
        b, c, h, w = conv_out.size()
        
        # שטיח את ערוץ הגובה והערוצים יחד כדי שנוכל להעביר ל-RNN
        conv_out = conv_out.view(b, c * h, w) # צורה חדשה: [Batch_Size, 1536, Width]
        
        # RNN ב-PyTorch מצפה לקבל את הרצף במימד הראשון: [Sequence_Length, Batch_Size, Features]
        conv_out = conv_out.permute(2, 0, 1) 
        
        # העברה ב-RNN
        rnn_out, _ = self.rnn(conv_out)
        
        # העברה בשכבת הפלט לקבלת הסתברויות לכל אות
        output = self.fc(rnn_out)
        
        # מחזירים log_softmax - זו דרישת חובה של אלגוריתם ה-CTC Loss ב-PyTorch
        return nn.functional.log_softmax(output, dim=2)