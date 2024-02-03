# Analiza Sentymentu w Finansowych Nagłówkach Prasowych

Projekt NLP demonstruje użycie głębokiego uczenia do analizy sentymentu w nagłówkach prasowych dotyczących finansów, zebranych z CNBC, wykorzystując CNN z osadzeniami GloVe i pre-trenowany model BERT. Porównanie obejmuje również użycie Rekurencyjnej Sieci Neuronowej LSTM.

## Wymagania

- Python 3.6.0+
- PyTorch
- TensorFlow
- Transformers
- Datasets
- Keras
- Pandas
- Numpy
- Matplotlib
- Seaborn

## Instalacja


## Przygotowanie Danych

Dane wejściowe w formacie CSV zawierają dwie kolumny: Sentiment oraz Headline, gdzie Sentiment to etykiety sentymentu (pozytywny, neutralny, negatywny), a Headline to tekst nagłówków.

## Model BERT

Wykorzystanie pre-trenowanego modelu BERT (distilbert-base-uncased) do klasyfikacji sentymentu, finetunowanego na danych z nagłówków finansowych CNBC.

## Trening Modelu BERT

Użycie trainera z biblioteki transformers, z konfiguracją treningu zawartą w `TrainingArguments`.

## Model CNN z GloVe

Model CNN z osadzeniami GloVe jako pierwsza warstwa w modelu analizy sentymentu, wykorzystujący warstwy konwolucyjne do ekstrakcji cech z tekstu.

## Trening Modelu CNN

Model CNN trenowany z użyciem Kerasa, z osadzeniami GloVe do inicjalizacji warstwy osadzeń.

## Ocena Sentymentu

Funkcje `predict` dla modeli BERT i CNN pozwalają na przewidywanie sentymentu nowych nagłówków prasowych.

## Przykłady Użycia

```python
text = "Meta shares up 20% as investors cheer dividends and $50bn buyback"
# Dla BERTa
sentiment_bert = predict(text, bert_model, tokenizer):
# Dla pozostałych modeli (np. tu dla CNN)
sentiment_cnn = predict_other(text, model_cnn_ae, tokenizer, max_len=128):
print(f"Sentiment (BERT): {sentiment_bert}")
print(f"Sentiment (CNN): {sentiment_cnn}")
```

## Autor

Kacper Kresa s29563
