# NATGAS Market Analyzer

Ten projekt to narzędzie do automatycznej analizy rynku gazu ziemnego (NATGAS), wykorzystujące dane:
- techniczne (Alpha Vantage),
- fundamentalne (EIA),
- pogodowe (OpenWeatherMap),
- newsowe (NewsAPI),
- oraz generujące raport końcowy za pomocą GPT-4 (OpenAI API).

## Funkcjonalności

- Pobieranie i cache’owanie danych rynkowych z Alpha Vantage (interwały minutowe i dzienne)
- Obliczanie wskaźników technicznych (RSI, MACD, ATR, VWAP, Pivot Points itd.)
- Pobieranie najnowszych newsów z NewsAPI na podstawie ponad 40 zapytań tematycznych
- Pobieranie prognozy pogody z OpenWeatherMap (np. Texas, USA)
- Pobieranie danych fundamentalnych (ceny i magazynowanie gazu) z EIA API
- Generowanie raportu z analizą i strategią handlową na kolejny dzień przy użyciu GPT-4

## Wymagania

- Python 3.8+
- Klucze API:
  - Alpha Vantage
  - NewsAPI
  - OpenWeatherMap
  - OpenAI
  - EIA

## Instalacja

```bash
pip install -r requirements.txt
```

Plik `requirements.txt` powinien zawierać:
```
requests
pandas
alpha_vantage
pandas_ta
openai
scikit-learn
matplotlib
mplfinance
```

## Użycie

Uruchom główny plik:

```bash
python natgas.py
```

Zostaną:
- pobrane dane z API,
- wykonana analiza techniczna,
- wygenerowany raport do pliku `daily_report.txt`.

## Uwaga

W kodzie znajdują się placeholdery `"API"` – uzupełnij je swoimi kluczami API przed użyciem.

## Pliki wyjściowe

- `market_data_<ticker>_<interval>.csv` – dane techniczne
- `news_articles.csv` – newsy z NewsAPI
- `daily_report.txt` – raport analizy z GPT-4

