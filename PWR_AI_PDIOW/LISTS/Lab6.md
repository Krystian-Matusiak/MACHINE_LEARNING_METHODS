# Zadania Lista 6

## 1. (3 pkt) Tuning hiperparametrów

Porównaj następujące metody optymalizacji parametrów modelu: _RandomSearch, GridSearch, HalvingGridSearch, HalvingRandomSearch_

a) (0.5 pkt) Opisz pokrótce ich działanie, możliwości i ograniczenia

b) (1.5 pkt) Użyj ich i porównaj jakość najlepszego modelu oraz czas ich działania. Narysuj odpowiedni wykres.

c) (0.5 pkt) Podaj lub oszacuj ilość iteracji każdej z metod (tj. liczbę stworzonych modeli podczas przeszukania)

d) (0.5 pkt) Podsumuj uzyskane wyniki. Według własnego uznania wskaż najlepszą wykorzystaną metodę optymalizacji hiperparametrów. Wybór uzasadnij.

## 2. (4 pkt) Wektoryzacja tekstu

a) (1 pkt) Używając biblioteki Spacy, dokonaj preprocessingu danych tekstowych. Usuń z tekstu wszystkie elementy, które nie są nośnikami emocji. 

b) (3 pkt) Dokonaj wektoryzacji tak przetworzonego tekstu przy pomocy następujących metod:
- bag-of-words
- tf-idf
- word2vec

Wyucz klasyfikator używając każdej z nich. Porównaj wyniki i spróbuj uzasadnić różnice w jakości. 

## 3. (3 pkt) Wyjaśnialność modeli

a) (1 pkt) Używając metody SHAP, przeanalizuj wyniki jednego z uprzednio stworzonych klasyfikatorów. 

b) (1 pkt) Zweryfikuj, które cechy wpływają na ocenę sentymentu pozytywnie, a które negatywnie. 

c) (1 pkt) Przeanalizuj dokładnie kilka błędnych zaklasyfikoawanych przykładów, sprawdzając, co w ich przypadku spowodowało błąd.
