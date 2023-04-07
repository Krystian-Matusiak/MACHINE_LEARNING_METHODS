# Zadania Lista 3


## Wymagania

- Komendy `dvc` należy uruchomić korzystając ze środowiska w kontenerze Docker, w przeciwnym razie przyznane będzie 0pkt za zadania z DVC. 
- MLflow należy uruchomić w Dockerze, w przeciwnym razie przyznane będzie 0pkt za zadania z MLflow.

---

## Analiza wydźwięku

Będziemy zajmować się zadaniem analizy wydźwięku emocjonalnego (sentiment).

Analiza wydźwięku (ang. sentiment analysis) będzie polegać na określeniu nacechowania emocjonalnego recenzji internetowych.

Jest to rozbudowany i mocno subiektywny problem - tekst ma inny wydźwięk dla autora tekstu, inny dla osoby czytającej, inny dla “podmiotu lirycznego”; mocno zależy też od kontekstu kulturowego.

Najczęstszym przypadkiem jest uproszczenie zadania do trójklasowej klasyfikacji - podziału na teksty pozytywne, neutralne, negatywne.

**Żródło danych**

Dzisiejsze zadania rozpoczynają przygodę ze zbiorem danych, którą kontynuować będziemy przez najbliższe laboratoria. **Zadbaj o jakość i reprodukowalność kodu.** 

Będziemy korzystać ze zbioru danych Amazon Review Data, dokumentację znajdziemy [tutaj](https://nijianmo.github.io/amazon/index.html):
* grupa 1 i 2 (czw. 15:15 i pt. 9:15):
    - https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Luxury_Beauty_5.json.gz
* grupa 3 i 4 (pt. 11:15 i pt. 15:15):

    (pliki należy połączyć w ramach wstępnego przetwarzania danych, a kategorię wpisać jako nową kolumnę)
    - https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz
    - https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz
    - https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Appliances_5.json.gz
    - https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Software_5.json.gz

**Etykiety**
* grupa 1 i 2: kolumna "overall", uproszczona następująco (do wyboru): 1,2 -> "negative", 3 -> "neutral", 4,5 -> "positive" **lub** 1,2 -> "negative", 3, 4 -> "neutral", 5 -> "positive"
* grupa 3 i 4: kolumna "overall"

---

## 1. (3 pkt) Przygotowanie potoku przetwarzaniadanych

Przygotuj poszczególne elementy rozwiązania problemu uczenia maszynowego w postaci osobnych skryptów.

Do realizacji zastosować następujący podział:
- Wstępne przetwarzanie danych
- Podział danych
- Analiza danych
- Ekstrakcja cech
- Uczenie i ewaluacja modelu

Skrypty umieść w folderze `scripts`, parametry w pliku `params.yaml`, a pliki wejściowe i wyjściowe w folderze `data` (najlepiej utwórz tam odpowiednie podfoldery). (Struktura projektu taka jak na laboratorium 1.)

W ramach tej listy zajmiemy się budową klasyfikatora, który będzie punktem odniesienia (baseline) w kolejnych listach. Będzie on predykował ignorując wszystkie cechy wejściowe. Ponieżej zamieszczono opis działania poszczególnych skryptów.

- Wstępne przetwarzanie danych - w razie konieczności połącz pliki oraz rozwiąż problematyczny zagnieżdżony atrybut `style` i zapisz wynik jako plik/i.
- Podział danych - wydziel zbiór testowy bez stratyfikacji i oba zbiory zapisz. Zaleca się wykorzystanie funkcji [sklearn.model_selection.train_test_split
](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
- Ekstrakcja cech - usuń wszystkie cechy (poza klasą sentymentu) oraz dodaj nową kolumnę, którą całą wypełnisz wartością `None`. Zapisz wynik.
- Analiza danych - na razie pomijamy, zajmiemy się na przyszłych zajęciach.
- Uczenie i ewaluacja modelu - wytrenuj klasyfikator który będzie predykował najczęstszą klasę, następnie policz F1-score na zbiorze testowym, wyniki zapisz w formacie json (lub podobnym obsługiwanym przez dvc w kontekście metryk). Zaleca się wykorzystanie [sklearn.dummy.DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) oraz [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).


**Uwagi**
* należy dużą uwagę przyłożyć do tego w których skryptach umieścić poszczególne funkcjonalności
* należy dobrze się zastanowić jakie parametry umieścić w `params.yaml`
* decyzje nie zawsze będą proste i czasem będzie istniało więcej niż jedno rozwiązanie, należy jednak zdawać sobie sprawę z wad i zalet danej implementacji 


## 2. (3 pkt)  Obsługa DVC

* Dodaj przygotowane skrypty jako elementy potoku `DVC`, zdefiniuj odpowiednie parametry, zależności oraz wyjścia.
* Dodaj metryki do trackowania.
* Metryk nie należy umieszczać w cache dvc, mają być trackowane przez git. [więcej tutaj](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#metrics-and-plots-outputs) 
* Zapisz wynik komendy `dvc status` (zrzut ekranu).
* Zreprodukuj potok. Zatwierdź zmiany i umieść je w repozytorium. 
* Zmień podział danych na stratyfikowany. Zapisz wynik komendy `dvc status` (zrzut ekranu). Zreprodukuj potok. Zatwierdź zmiany i umieść je w repozytorium. 

## 3. (3 pkt) Wykonanie eksperymentów

* Dodaj wsparcie dla modelu predykującego klasę z rozkładu jednostajnego ciągłego. Dodaj możliwość wybrania modelu w konfiguracji (params.yaml). 
* Wykorzystaj api `dvc experiments` do porównania modeli. 
* Załącz wyjście polecenia dvc experiments show (zrzut ekranu).


## 4. (3 pkt) Obsługa MLflow

* Do skryptu przeznaczonego do ewaluacji modelu dodaj obsługę `MLflow`. 
* Przekaż odpowiednie parametry i metryki. 
* Dodaj wykres macierzy pomyłek dla zbioru uczącego i testowego.
* Załącz zrzuty ekranu z MLFlow pokazujące przebieg eksperymentów.

## 5. [OPCJONALNIE] (1 pkt) DVC remote

* Dodaj zewnętrzną lokację do przechowywania danych z DVC np. studencki Google Drive.
* Prześlij dane.
