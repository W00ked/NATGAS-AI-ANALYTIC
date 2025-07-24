
import requests
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import pandas_ta as ta
from datetime import datetime, timedelta
import numpy as np
import openai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import mplfinance as mpf
import time
import os


# Pobieranie danych rynkowych z Yahoo Finance (dziennych)
def fetch_market_data(ticker, interval="15min", output_size="full", retries=5, delay=60):
    """
    Pobiera dane rynkowe z Alpha Vantage dla podanego symbolu i interwału czasowego.
    Obsługuje błędy API, zapisuje dane do plików CSV (cache) i dodaje opóźnienia, jeśli są limity zapytań.

    :param ticker: Symbol giełdowy (np. "AAPL", "NG=F").
    :param interval: Interwał czasowy (np. "1min", "5min", "15min", "30min", "60min", "daily").
    :param output_size: "full" (pełna historia) lub "compact" (100 ostatnich wartości).
    :param retries: Liczba prób pobrania danych w razie błędu.
    :param delay: Opóźnienie między próbami w sekundach.

    :return: DataFrame z danymi rynkowymi lub None, jeśli pobranie nie powiodło się.
    """
    api_key = "JITM9VUPTGM2RFEW"
    filename = f"market_data_{ticker}_{interval}.csv"

    # 1. Sprawdzamy, czy dane już istnieją (cache)
    if os.path.exists(filename):
        print(f"📄 Dane dla {ticker} ({interval}) już istnieją. Wczytuję z pliku.")
        return pd.read_csv(filename, index_col=0, parse_dates=True)

    print(f"Pobieranie danych rynkowych dla: {ticker}, interwał: {interval}...")

    ts = TimeSeries(key=api_key, output_format="pandas")

    for attempt in range(retries):
        try:
            print(f"Próba {attempt + 1} pobierania danych dla {ticker} ({interval})...")

            if "min" in interval:
                data, meta_data = ts.get_intraday(symbol=ticker, interval=interval, outputsize=output_size)
            else:
                data, meta_data = ts.get_daily(symbol=ticker, outputsize=output_size)

            if data.empty:
                print(f"Pobranie zakończone, ale brak danych dla {ticker} w {interval}.")
                return None

            # 3. Formatowanie kolumn
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            data.index = pd.to_datetime(data.index)
            data.sort_index(inplace=True)

            # 4. Zapisywanie danych do pliku (cache)
            data.to_csv(filename)
            print(f" Pobranie zakończone sukcesem. Dane zapisane do {filename}.")
            return data

        except Exception as e:
            print(f" Błąd pobierania danych: {e}")
            print(f" Czekam {delay} sekund przed kolejną próbą...")
            time.sleep(delay)

    print(f"Nie udało się pobrać danych dla {ticker} po {retries} próbach.")
    return None



def calculate_pivot_points(data):
    """
    Oblicza klasyczne poziomy Pivot Points na podstawie danych OHLC.
    """
    if 'High' not in data.columns or 'Low' not in data.columns or 'Close' not in data.columns:
        print(" Brak wymaganych kolumn do obliczenia Pivot Points.")
        return data  # Zwraca oryginalne dane, jeśli czegoś brakuje

    print(" Obliczanie Pivot Points...")

    data['Pivot_Point'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['Support_1'] = (2 * data['Pivot_Point']) - data['High']
    data['Resistance_1'] = (2 * data['Pivot_Point']) - data['Low']
    data['Support_2'] = data['Pivot_Point'] - (data['High'] - data['Low'])
    data['Resistance_2'] = data['Pivot_Point'] + (data['High'] - data['Low'])

    print(" Pivot Points obliczone.")
    return data



import pandas_ta as ta

def add_technical_indicators(data):
    """
    Dodaje wskaźniki techniczne do danych rynkowych i obsługuje błędy związane z NaN oraz formatem zwracanych wyników.
    """
    if data is None or data.empty:
        print(" Brak danych rynkowych, wskaźniki techniczne nie zostaną dodane.")
        return None

    print(" Dodawanie wskaźników technicznych...")

    required_columns = ['Close', 'High', 'Low', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"❌ Brak wymaganych kolumn: {', '.join(missing_columns)}")
        return None

    def check_nan(df, indicator_name):
        """Sprawdza, czy wskaźnik zawiera NaN i informuje użytkownika."""
        if indicator_name in df.columns:
            nan_count = df[indicator_name].isna().sum()
            if nan_count > 0:
                print(f" Ostrzeżenie: Wskaźnik {indicator_name} zawiera {nan_count} brakujących wartości.")

    try:
        # RSI (Relative Strength Index)
        data['RSI_14'] = ta.rsi(data['Close'], length=14)
        check_nan(data, 'RSI_14')

        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
        if isinstance(macd, pd.DataFrame):
            data['MACD'] = macd.iloc[:, 0]
            data['MACD_signal'] = macd.iloc[:, 1]
            data['MACD_hist'] = macd.iloc[:, 2]
            check_nan(data, 'MACD')
            check_nan(data, 'MACD_signal')
            check_nan(data, 'MACD_hist')

        # EMA (Exponential Moving Average)
        data['EMA_10'] = ta.ema(data['Close'], length=10)
        data['EMA_50'] = ta.ema(data['Close'], length=50)
        check_nan(data, 'EMA_10')
        check_nan(data, 'EMA_50')

        # SMA (Simple Moving Average)
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        check_nan(data, 'SMA_20')
        check_nan(data, 'SMA_50')

        # WMA (Weighted Moving Average)
        data['WMA_20'] = ta.wma(data['Close'], length=20)
        check_nan(data, 'WMA_20')

        # ADX (Average Directional Index)
        adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
        if isinstance(adx, pd.DataFrame):
            data['ADX_14'] = adx.iloc[:, 0]
            check_nan(data, 'ADX_14')

        # CCI (Commodity Channel Index)
        data['CCI_14'] = ta.cci(data['High'], data['Low'], data['Close'], length=14)
        check_nan(data, 'CCI_14')

        # Stochastic Oscillator (STOCH)
        stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=14, d=3)
        if isinstance(stoch, pd.DataFrame):
            data['Stoch_K'] = stoch.iloc[:, 0]
            data['Stoch_D'] = stoch.iloc[:, 1]
            check_nan(data, 'Stoch_K')
            check_nan(data, 'Stoch_D')

        # Ichimoku Cloud
        ichimoku = ta.ichimoku(data['High'], data['Low'], data['Close'])
        if isinstance(ichimoku, tuple):
            ichimoku = ichimoku[0]
        if isinstance(ichimoku, pd.DataFrame):
            data['Ichimoku_Base'] = ichimoku.iloc[:, 0]
            data['Ichimoku_Conversion'] = ichimoku.iloc[:, 1]
            check_nan(data, 'Ichimoku_Base')
            check_nan(data, 'Ichimoku_Conversion')

        # Momentum Indicator (MOM)
        data['MOM_10'] = ta.mom(data['Close'], length=10)
        check_nan(data, 'MOM_10')

        # Williams %R (WILLR)
        data['WILLR_14'] = ta.willr(data['High'], data['Low'], data['Close'], length=14)
        check_nan(data, 'WILLR_14')

        # Rate of Change (ROC)
        data['ROC_10'] = ta.roc(data['Close'], length=10)
        check_nan(data, 'ROC_10')

        # Chaikin Money Flow (CMF)
        data['CMF_14'] = ta.cmf(data['High'], data['Low'], data['Close'], data['Volume'], length=14)
        check_nan(data, 'CMF_14')

        # VWAP (Volume Weighted Average Price)
        data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        check_nan(data, 'VWAP')

        # Bollinger Bands
        bollinger = ta.bbands(data['Close'], length=20)
        if isinstance(bollinger, pd.DataFrame):
            data['BB_upper'] = bollinger.iloc[:, 0]
            data['BB_middle'] = bollinger.iloc[:, 1]
            data['BB_lower'] = bollinger.iloc[:, 2]
            check_nan(data, 'BB_upper')
            check_nan(data, 'BB_middle')
            check_nan(data, 'BB_lower')

        # ATR (Average True Range)
        data['ATR_14'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        check_nan(data, 'ATR_14')

        # Pivot Points
        data = calculate_pivot_points(data)

        # Obsługa brakujących wartości
        missing_before = data.isna().sum().sum()
        if missing_before > 0:
            print(f"Przed uzupełnieniem: {missing_before} brakujących wartości.")

        data.bfill(inplace=True)
        data.ffill(inplace=True)
        data.fillna(0, inplace=True)

        missing_after = data.isna().sum().sum()
        if missing_after == 0:
            print(" Wszystkie brakujące wartości zostały uzupełnione.")
        else:
            print(f" Po uzupełnieniu nadal {missing_after} brakujących wartości.")

        print(" Dodano wskaźniki techniczne.")
        return data

    except Exception as e:
        print(f" Błąd podczas dodawania wskaźników technicznych: {e}")
        return None




# Pobieranie danych z NewsAPI
def fetch_daily_news(api_key, queries, from_date, to_date):
    """
    Pobiera artykuły z NewsAPI dla wielu fraz wyszukiwania w określonym zakresie dat.
    :param api_key: Klucz API NewsAPI.
    :param queries: Lista fraz do wyszukiwania.
    :param from_date: Początkowa data w formacie YYYY-MM-DD.
    :param to_date: Końcowa data w formacie YYYY-MM-DD.
    :return: DataFrame z połączonymi wynikami dla wszystkich fraz.
    """
    print(f"Pobieranie artykułów z NewsAPI dla zakresu dat: {from_date} - {to_date}...")
    all_articles = []

    for query in queries:
        print(f"Pobieranie artykułów dla frazy: '{query}'...")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "apiKey": api_key,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            for article in articles:
                article["query"] = query  # Dodaj informację o frazie wyszukiwania
            all_articles.extend(articles)
            print(f"Pobrano {len(articles)} artykułów dla frazy '{query}'.")
        else:
            print(f"Błąd w pobieraniu artykułów dla frazy '{query}': {response.status_code}")

    if all_articles:
        return pd.DataFrame(all_articles)
    else:
        print("Nie znaleziono artykułów dla żadnej frazy.")
        return pd.DataFrame()



def fetch_weather_data(api_key, location="Texas,US"):
    """
    Pobiera prognozę pogody dla określonego regionu w USA.
    :param api_key: Klucz API OpenWeatherMap.
    :param location: Lokalizacja (domyślnie Texas, USA).
    :return: DataFrame z prognozami pogody.
    """
    print(f"Pobieranie prognozy pogody dla lokalizacji: {location}...")
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": location,
        "units": "metric",  # Możesz zmienić na 'imperial', jeśli wolisz °F
        "appid": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Parsowanie istotnych informacji
        forecasts = [
            {
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "weather": item["weather"][0]["description"],
                "wind_speed": item["wind"]["speed"]
            }
            for item in data["list"]
        ]
        return pd.DataFrame(forecasts)
    else:
        print(f"Błąd w pobieraniu danych pogodowych: {response.status_code}")
        return None


# Pobieranie najnowszych danych z EIA
def fetch_latest_eia_data(api_key, endpoint, params):
    """
    Pobiera najnowsze dane z EIA API na podstawie klucza API i parametrów.
    """
    base_url = "https://api.eia.gov/v2"
    url = f"{base_url}{endpoint}"

    # Dodaj klucz API do parametrów
    params["api_key"] = api_key

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "response" in data and "data" in data["response"]:
            df = pd.DataFrame(data["response"]["data"])
            return df.sort_values(by="period", ascending=False).head(1)  # Najnowszy rekord
        else:
            print("Brak danych w odpowiedzi.")
            return None
    else:
        print(f"Błąd w pobieraniu danych: {response.status_code}")
        return None


def analyze_on_intervals(ticker, start_date, end_date, intervals):
    """
    Przeprowadza analizę techniczną na różnych interwałach czasowych.
    """
    for interval in intervals:
        print(f"\n=== Analiza dla interwału: {interval} ===")
        market_data = fetch_market_data(ticker, start_date, end_date, interval=interval)
        if market_data is not None:
            market_data = add_technical_indicators(market_data)
            # Zapisz dane do pliku CSV
            csv_filename = f"market_data_{interval}.csv"
            market_data.to_csv(csv_filename)
            print(f"Dane techniczne dla interwału {interval} zapisane do '{csv_filename}'")


import openai


def analyze_with_gpt4(news_articles, eia_data, market_data, weather_data, openai_api_key):
    """
    Analizuje dane rynkowe, fundamentalne i techniczne przy użyciu GPT-4.
    """

    openai.api_key = openai_api_key

    openai.api_key = openai_api_key

    # Sprawdź, czy `news_articles` to DataFrame, jeśli nie, zamień na pusty DataFrame
    if not isinstance(news_articles, pd.DataFrame):
        print("⚠️ Błąd: `news_articles` to dict, zamieniam na pusty DataFrame.")
        news_articles = pd.DataFrame()

    # Sprawdź `eia_data` (może to być dict!)
    if isinstance(eia_data, dict):
        for key, value in eia_data.items():
            if not isinstance(value, pd.DataFrame):  # Jeśli nie jest DataFrame, zamień na pusty
                print(f"⚠️ `eia_data[{key}]` to dict, zamieniam na pusty DataFrame.")
                eia_data[key] = pd.DataFrame()

    # Sprawdź `market_data` (może to być dict!)
    if isinstance(market_data, dict):
        for key, value in market_data.items():
            if not isinstance(value, pd.DataFrame):  # Jeśli nie jest DataFrame, zamień na pusty
                print(f"⚠️ `market_data[{key}]` to dict, zamieniam na pusty DataFrame.")
                market_data[key] = pd.DataFrame()

    # Sprawdź `weather_data`, jeśli nie jest DataFrame, zamień na pusty
    if not isinstance(weather_data, pd.DataFrame):
        print("⚠️ Błąd: `weather_data` to dict, zamieniam na pusty DataFrame.")
        weather_data = pd.DataFrame()

    # Przygotowanie podsumowania wiadomości rynkowych
    news_summary = "Brak dostępnych artykułów."
    if not news_articles.empty:
        news_summary = "\n".join(
            [f"- {row['title']} ({row['publishedAt']})" for _, row in news_articles.iterrows()]
        )

    # Przygotowanie podsumowania danych fundamentalnych (EIA)
    eia_summary = "\n".join([
        f"{category}: {df.to_string(index=False)}" if isinstance(df, pd.DataFrame) and not df.empty else f"{category}: Brak danych"
        for category, df in eia_data.items()
    ])

    # Przygotowanie podsumowania pogody
    weather_summary = "Brak danych pogodowych."
    if not weather_data.empty:
        weather_summary = weather_data.head(5).to_string(index=False)

    # Pobranie najnowszych wskaźników technicznych
    def safe_format(value):
        """Konwertuje wartość na liczbę zmiennoprzecinkową, jeśli to możliwe, inaczej zwraca tekst"""
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return "Brak"


    if isinstance(market_data, dict) and market_data:  # Sprawdzamy, czy market_data nie jest pusty
        latest_interval = list(market_data.keys())[-1]  # Pobieramy ostatni dostępny interwał
        latest_data = market_data[latest_interval].iloc[-1]  # Pobieramy ostatni wiersz danych

    tech_summary = (
        f" **Wskaźniki techniczne ({latest_interval}):**\n"
        f"- RSI_14: {safe_format(latest_data.get('RSI_14', 'Brak'))}\n"
        f"- MACD: {safe_format(latest_data.get('MACD', 'Brak'))}, MACD Signal: {safe_format(latest_data.get('MACD_signal', 'Brak'))}\n"
        f"- ATR_14: {safe_format(latest_data.get('ATR_14', 'Brak'))}\n"
        f"- Pivot Point: {safe_format(latest_data.get('Pivot_Point', 'Brak'))}\n"
        f"- Support_1: {safe_format(latest_data.get('Support_1', 'Brak'))}, Resistance_1: {safe_format(latest_data.get('Resistance_1', 'Brak'))}\n"
        f"- VWAP: {safe_format(latest_data.get('VWAP', 'Brak'))}\n"
    )

    # Tworzenie promptu dla GPT-4
    prompt = f"""
        Jesteś profesjonalnym analitykiem rynku surowców specjalizującym się w gazie ziemnym (NATGAS). 
        Twoim zadaniem jest przeprowadzenie kompleksowej analizy rynku na podstawie danych technicznych, fundamentalnych oraz makroekonomicznych. 
        Przedstaw precyzyjne prognozy i zalecenia inwestycyjne.

        **Najważniejsze wiadomości rynkowe**:
        {news_summary}

        **Dane fundamentalne (EIA)**:
        {eia_summary}

        **Prognoza pogody**:
        {weather_summary}

        {tech_summary}

        **Analiza techniczna na interwałach 15m, 30m, 1H, 1D**:
        - Czy cena jest w trendzie wzrostowym, spadkowym czy konsolidacji?
        - Jakie są kluczowe poziomy wsparcia i oporu na podstawie Price Action oraz Fibonacci?
        - Jakie formacje świecowe dominują (np. engulfing, doji, pin bar)?
        - Czy RSI sugeruje wykupienie (>70) lub wyprzedanie (<30) rynku?
        - Czy MACD generuje sygnał kupna/sprzedaży? Jak zachowuje się histogram?
        - Czy zmienność (ATR) rośnie, co może wskazywać na większy ruch cenowy?
        - Czy ADX potwierdza siłę trendu (wartość >25)?
        - Jakie poziomy Pivot Points oraz VWAP mogą być kluczowe dla day tradingu?
        - Czy wolumen potwierdza ruchy cenowe (rośnie przy wzrostach/spadkach)?

         **Prognoza i strategia tradingowa**:
        - Scenariusz optymistyczny (wzrost cen) – jakie warunki muszą zostać spełnione?
        - Scenariusz neutralny (stabilizacja rynku) – jakie sygnały sugerują konsolidację?
        - Scenariusz pesymistyczny (spadek cen) – jakie wskaźniki potwierdzają dalsze spadki?

        **Strategia na kolejny dzień**:
        - Rekomendowane poziomy wejścia i wyjścia:
          - Punkt wejścia (Buy/Sell): 
          - Take Profit (TP): 
          - Stop Loss (SL): 
          - Risk-Reward Ratio (RRR): 

        - Jakie kluczowe wydarzenia mogą wpłynąć na zmienność rynku?
        - Jakie dane makro mogą wpłynąć na rynek w nadchodzących dniach?

        Podaj finalne wnioski oraz zalecaną strategię na następny dzień.
    """

    # Wysłanie zapytania do GPT-4
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem od analizy technicznej rynków surowcowych."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']

    except Exception as e:
        print(f" Błąd analizy GPT-4: {e}")
        return "Nie udało się wygenerować raportu."


# Główna funkcja
if __name__ == "__main__":
    # Pobieranie zakresu dat
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")  # Ostatnie 30 dni

    # NewsAPI - Pobieranie wiadomości rynkowych
    news_api_key = "API"
    news_queries = [
        "EIA natural gas storage report",
        "US natural gas inventories",
        "Weekly gas storage report",
        "LNG exports US",
        "natural gas consumption trends",
        "natural gas demand forecast",
        "Henry Hub price forecast",
        "natural gas pipeline outages",
        "Russia gas supply Europe",
        "Nord Stream pipeline update",
        "Ukraine gas transit",
        "China LNG imports",
        "US-Europe LNG exports",
        "natural gas supply chain disruptions",
        "Middle East gas supply",
        "OPEC+ natural gas policies",
        "US shale production",
        "Permian Basin gas production",
        "natural gas drilling activity",
        "NOAA natural gas weather forecast",
        "US winter forecast energy demand",
        "heatwave impact on gas consumption",
        "polar vortex natural gas impact",
        "hurricane impact on Gulf Coast LNG",
        "El Niño La Niña effect on energy",
        "winter heating demand forecast",
        "summer cooling demand natural gas",
        "storage injections vs withdrawals",
        "EU natural gas price cap",
        "US energy policy natural gas",
        "EPA methane regulations",
        "Biden LNG export policy",
        "carbon tax impact on gas markets",
        "US GDP energy demand correlation",
        "interest rates impact on energy commodities",
        "inflation effect on energy markets",
        "currency impact on LNG exports",
        "natural gas price volatility",
        "options open interest Henry Hub",
        "CFTC natural gas futures positioning",
        "hedge funds natural gas positioning",
        "natural gas technical analysis trend",
        "LNG vs pipeline gas pricing"
    ]

    try:
        news_articles = fetch_daily_news(news_api_key, news_queries, from_date=yesterday, to_date=today)
        if not news_articles.empty:
            print(f" Znaleziono {len(news_articles)} artykułów rynkowych.")
            news_articles.to_csv("news_articles.csv", index=False)
        else:
            print("Nie znaleziono żadnych artykułów.")
    except Exception as e:
        print(f" Błąd pobierania newsów: {e}")
        news_articles = pd.DataFrame()

    # Pobieranie danych fundamentalnych z EIA
    eia_api_key = "1hR0HSsvUqtozueLzI40Ec0JdChx6Ea5Fi7uKzNF"
    eia_requests = {
        "Prices": {
            "endpoint": "/natural-gas/pri/sum/data",
            "params": {"frequency": "monthly", "data[0]": "value",
                       "sort[0][column]": "period", "sort[0][direction]": "desc",
                       "offset": 0, "length": 1}
        },
        "Storage": {
            "endpoint": "/natural-gas/stor/wkly/data",
            "params": {"frequency": "weekly", "data[0]": "value",
                       "sort[0][column]": "period", "sort[0][direction]": "desc",
                       "offset": 0, "length": 1}
        }
    }

    eia_data = {}
    for category, request in eia_requests.items():
        try:
            eia_data[category] = fetch_latest_eia_data(eia_api_key, request["endpoint"], request["params"])
        except Exception as e:
            print(f" Błąd pobierania danych EIA ({category}): {e}")
            eia_data[category] = None

    # Pobieranie i analiza danych rynkowych
    ticker = "NG"
    intervals = ["15min"] # "30min", "60min", "daily"

    market_data = {}
    for interval in intervals:
        print(f"\n=== 📊 Analiza techniczna dla interwału: {interval} ===")
        try:
            data = fetch_market_data(ticker, interval=interval)
            if data is not None and not data.empty:
                data = add_technical_indicators(data)
                csv_filename = f"market_data_{ticker}_{interval}.csv"
                data.to_csv(csv_filename)
                market_data[interval] = data
                print(f" Dane zapisane do '{csv_filename}'")
            else:
                print(f"Brak danych dla interwału {interval}.")
        except Exception as e:
            print(f" Błąd analizy technicznej dla {interval}: {e}")

    # Pobieranie danych pogodowych
    weather_api_key = "API"
    try:
        weather_data = fetch_weather_data(weather_api_key, location="Texas,US")
        if weather_data is not None:
            print("\n🌤 Prognoza pogody dla Teksasu:")
            print(weather_data.head(5).to_string(index=False))
        else:
            print("Nie udało się pobrać prognozy pogody.")
    except Exception as e:
        print(f" Błąd pobierania danych pogodowych: {e}")
        weather_data = None

    # Analiza z GPT-4
    openai_api_key = "API"
    try:
        daily_report = analyze_with_gpt4(news_articles, eia_data, market_data, weather_data, openai_api_key)
        print("\n📜 Raport GPT-4:")
        print(daily_report)

        # Zapis raportu do pliku
        with open("daily_report.txt", "w", encoding="utf-8") as file:
            file.write(daily_report)
        print("✅ Raport zapisany do 'daily_report.txt'.")
    except Exception as e:
        print(f"❌ Błąd analizy GPT-4: {e}")
