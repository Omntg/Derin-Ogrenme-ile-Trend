# main.py

import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model

#--------------------------------------------------------------------
# 1) FastAPI uygulaması ve template klasörünü ayarla
#--------------------------------------------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# NOT: "static" klasörünüz olmadığı için app.mount satırı kaldırıldı.

#--------------------------------------------------------------------
# 2) Model ve Scaler'ları yükle (uygulama başlarken bir defa)
#--------------------------------------------------------------------
model = load_model("best_trend_model_reg.h5")    # Eğittiğiniz L2‐düzenleyicili model

scaler_ema     = joblib.load("scaler_ema.pkl")
scaler_kalman  = joblib.load("scaler_kalman.pkl")
scaler_kama    = joblib.load("scaler_kama.pkl")
scaler_hlcv    = joblib.load("scaler_hlcv.pkl")

#--------------------------------------------------------------------
# 3) Excel'den tam veri setini yükle
#--------------------------------------------------------------------
df_all = pd.read_excel("features_labeled.xlsx", engine="openpyxl")
df_all["Date"] = pd.to_datetime(df_all["Date"])

# Dropdown menü için hisseler
unique_tickers = sorted(df_all["Stock"].unique().tolist())

# Model'in beklediği ham feature kolonları (toplam 18 sütun; eğitimde kullandığınız sıraya göre)
feature_cols = [
    "EMA_8", "EMA_13", "EMA_21", "EMA_34", "EMA_55", "EMA_89", "EMA_144",
    "kalman_short", "kalman_long",
    "kama_13", "kama_21", "kama_34", "kama_55",
    "High", "Low", "Close", "Volume", "XU100"
]

# Sliding‐window uzunluğu (eğitimle aynı 60)
SEQ_LEN = 60

#--------------------------------------------------------------------
# 4) "/" → Ana sayfa (form + boş grafik alanı)
#--------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Ana sayfa: Hisse seçim formu ve (henüz grafik yok) alanı.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tickers": unique_tickers
        }
    )

#--------------------------------------------------------------------
# 5) "/predict/" → Form gönderildiğinde seçimi al, aynı template'ı render et
#--------------------------------------------------------------------
@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request):
    """
    POST ile gelen form verisinden 'ticker' alıp, index.html içinde
    selected_ticker olarak döndürür. Grafiği JS tarafı çizecek.
    """
    form_data = await request.form()
    selected_ticker = form_data.get("ticker")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tickers": unique_tickers,
            "selected_ticker": selected_ticker
        }
    )

#--------------------------------------------------------------------
# 6) Ardışık aynı sinyalleri filtrele (yalnızca değişiklik anları kalsın)
#--------------------------------------------------------------------
def filter_consecutive_signals(predictions):
    """
    Gelen predictions listesinde arka arkaya aynı 'pred' değeri varsa
    yalnızca ilkini tutar, sonraki aynı 'pred' değerlerini atar.
    """
    if len(predictions) == 0:
        return []
    filtered = [predictions[0]]
    for i in range(1, len(predictions)):
        if predictions[i]["pred"] != predictions[i-1]["pred"]:
            filtered.append(predictions[i])
    return filtered

#--------------------------------------------------------------------
# 7) "/plot-data/{ticker}" → JSON olarak OHLC + sinyal döner
#--------------------------------------------------------------------
@app.get("/plot-data/{ticker}")
async def plot_data(ticker: str):
    """
    Seçilen hissenin verisini al, sliding‐window mantığıyla four‐input
    modele girdi oluştur, predict et, 
    hem tam OHLC (all_data) hem de filtrelenmiş sinyal (predictions) JSON döndür.
    """
    # 1) Seçilen ticker'ın DataFrame'ini al ve tarihe göre sırala
    df_stock = df_all[df_all["Stock"] == ticker].sort_values("Date").reset_index(drop=True)
    if df_stock.empty:
        return JSONResponse(status_code=404, content={"error": "Ticker bulunamadı."})

    n = len(df_stock)
    if n < SEQ_LEN + 1:
        return JSONResponse(status_code=400, content={"error": "Yeterli veri yok."})

    # 2) Tüm günlerin OHLC verisini hazırla (all_data)
    all_data = []
    for i in range(n):
        date_str = df_stock.loc[i, "Date"].strftime("%Y-%m-%d")
        open_price = float(df_stock.loc[i-1, "Close"]) if i > 0 else float(df_stock.loc[i, "Close"])
        high_price = float(df_stock.loc[i, "High"])
        low_price  = float(df_stock.loc[i, "Low"])
        close_price= float(df_stock.loc[i, "Close"])
        all_data.append({
            "date": date_str,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })

    # 3) Sliding‐window bazlı sinyal tahmini (predictions)
    predictions = []
    # Yeterli veri varsa (n >= SEQ_LEN)
    if n >= SEQ_LEN:
        # Ham X (n,18) matrisi
        X_all = df_stock[feature_cols].values.astype(np.float32)

        # Dört gruba böl (eğitimdeki sıraya göre sabit indeksler)
        X_ema_full    = X_all[:,  0:7]    # (EMA_8 … EMA_144)
        X_kalman_full = X_all[:,  7:9]    # (kalman_short, kalman_long)
        X_kama_full   = X_all[:,  9:13]   # (kama_13 … kama_55)
        X_hlcv_full   = X_all[:, 13:18]   # (High, Low, Close, Volume, XU100)

        # Scaler'larla dönüştür
        X_ema_scaled    = scaler_ema.transform(X_ema_full)
        X_kalman_scaled = scaler_kalman.transform(X_kalman_full)
        X_kama_scaled   = scaler_kama.transform(X_kama_full)
        X_hlcv_scaled   = scaler_hlcv.transform(X_hlcv_full)

        # Sliding window dizilerini oluştur
        seqs_ema, seqs_kalman, seqs_kama, seqs_hlcv = [], [], [], []
        for i in range(SEQ_LEN, n):
            start, end = i - SEQ_LEN, i
            seqs_ema.append(    X_ema_scaled[start:end, :] )
            seqs_kalman.append( X_kalman_scaled[start:end, :] )
            seqs_kama.append(   X_kama_scaled[start:end, :] )
            seqs_hlcv.append(   X_hlcv_scaled[start:end, :] )

        # NumPy dizilerine çevir (n_seq, 60, kanal_sayısı)
        X_ema_seq    = np.stack(seqs_ema, axis=0).astype(np.float32)    # (n_seq,60,7)
        X_kalman_seq = np.stack(seqs_kalman, axis=0).astype(np.float32) # (n_seq,60,2)
        X_kama_seq   = np.stack(seqs_kama, axis=0).astype(np.float32)   # (n_seq,60,4)
        X_hlcv_seq   = np.stack(seqs_hlcv, axis=0).astype(np.float32)   # (n_seq,60,5)

        # Model prediction
        y_prob = model.predict(
            {
                "inp_ema":    X_ema_seq,
                "inp_kalman": X_kalman_seq,
                "inp_kama":   X_kama_seq,
                "inp_hlcv":   X_hlcv_seq
            },
            verbose=0
        ).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        # Tahminleri {date, close, pred} formatına getir
        for seq_idx in range(len(y_pred)):
            day_index = SEQ_LEN + seq_idx
            date_str  = df_stock.loc[day_index, "Date"].strftime("%Y-%m-%d")
            close_pr  = float(df_stock.loc[day_index, "Close"])
            pred_lbl  = int(y_pred[seq_idx])
            predictions.append({
                "date": date_str,
                "close": close_pr,
                "pred": pred_lbl
            })

    # 4) Ardışık aynı sinyalleri filtrele
    filtered_predictions = filter_consecutive_signals(predictions)

    # 5) JSON olarak döndür
    return {
        "ticker": ticker,
        "ohlc_data": all_data,
        "predictions": filtered_predictions
    }
