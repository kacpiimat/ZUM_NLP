Analiza Sentymentu 
w Finansowych Nagłówkach Prasowych
Mój projekt NLP pokazuje użycie głębokiego uczenia do analizy sentymentu nagłówków prasowych z wiadomościami dotyczącymi finansów zebranymi z CNBC. Ten i podobne zbiory do analizy sentymentu można znaleźć na Kaggle. Wykorzystuję 2 podejścia: jedno oparte o konwolucyjne sieci neuronowe (CNN) z osadzeniami GloVe oraz drugie - pre-trenowany model BERT (Bidirectional Encoder Representations from Transformers). Dla porównania demonstruję również Dokładność (Accuracy) zastosowania Rekurencyjnej Sieci Neuronowej LSTM
Wymagania
•	Python 3.6.0+
•	PyTorch
•	TensorFlow
•	Transformers
•	Datasets
•	Keras
•	Pandas
•	Numpy
•	Matplotlib
•	Seaborn
Instalacja
Aby zainstalować niezbędne biblioteki, wykonaj następujące polecenie:
pip install torch tensorflow transformers datasets keras pandas numpy matplotlib seaborn 
Przygotowanie Danych
Dane wejściowe są w formacie CSV z dwiema kolumnami: Sentiment oraz Headline. Kolumna Sentiment zawiera etykiety sentymentu (positive, neutral, negative), a kolumna Headline zawiera tekst nagłówków.
Model BERT
W projekcie wykorzystuję pre-trenowany model BERT (distilbert-base-uncased) do klasyfikacji sentymentu. Model jest finetunowany na podstawie dostarczonych danych z nagłówków finansowych CNBC.
Trening Modelu BERT
Do treningu modelu BERT używam trainera z biblioteki transformers. Konfiguracja treningu zawarta jest w TrainingArguments.
Model CNN z GloVe
Alternatywnie, stosujemy model CNN z osadzeniami (embeddings) GloVe i wykorzystaję je jako pierwszą warstwę w modelu dla analizy sentymentu. Model ten wykorzystuje warstwy konwolucyjne do ekstrakcji cech z tekstu, które następnie są używane do klasyfikacji sentymentu.
Trening Modelu CNN
Model CNN trenuję Kerasem. Używam osadzeń GloVe do inicjalizacji warstwy osadzeń w modelu.
Ocena Sentymentu
Po wytrenowaniu modeli można ich użyć do przewidywania sentymentu nowych nagłówków prasowych. Dla modelu BERT oraz CNN dostarczam funkcje predict, które przyjmują tekst nagłówka jako wejście i zwracają przewidywany sentyment.
Przykład Użycia
text = ”Meta shares up 20% as investors cheer dividends and $50bn buyback" sentiment_bert =
predict_bert(text, bert_model, tokenizer) sentiment_cnn = predict_cnn(text, model_cnn_ae) 
print(f”Sentiment (BERT): {sentiment_bert}") print(f”Sentiment (CNN): {sentiment_cnn}")
Autor
Kacper Kresa s29563


![Uploading image.png…]()
