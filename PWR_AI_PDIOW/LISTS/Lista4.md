# Zadania Lista 4: Analiza eksploracyjna danych

**Wymagania**

- Końcową komendę `dvc repro` należy uruchomić korzystając ze środowiska w kontenerze Docker. 

### Zadanie 1 (1 pkt)

Skonstruuj co najmniej 3 dodatkowe cechy na podstawie danych oraz wiedzy dziedzinowej i umieść je jako osobne kolumny. Cechy nie powinny pochodzić wyłącznie z tekstu. Przykładem takiej cechy może być długość tekstu.

### Zadanie 2 (6 pkt)

Celem zadania jest dokonanie analizy eksploracyjnej (EDA) zadanego zbioru danych, pod kątem zastosowania do treningu modelu sentymentu, przy pomocy poznanych narzędzi. 

Na podstawie danych numerycznych/kategorycznych postaraj się odpowiedzieć na następujące pytania (jednak nie ograniczaj się jedynie do odpowiedzi na nie):
- ile znajduje się w zbiorze cech kategorycznych, a ile numerycznych? 
- czy zmienna wyjściowa jest kategoryczna, czy numeryczna? 
- czy i ile w zbiorze jest brakujących wartości? Dla jakich zmiennych? Co z tego wynika?
- czy któreś z cech są skorelowane? Co z tego może wynikać?
- czy któraś z cech koreluje ze zmienną wyjściową? Jeśli tak - która? Czy któraś nie koreluje?
- czy zbiór danych wydaje się być wystarczająco informacyjny by rozwiązać zadanie analizy sentymentu?

Zbadaj również dane tekstowe:
- czy któreś ze słów wydają się dominować w zbiorze?
- czy najpopularniejsze słowa różnią się znacząco pomiędzy klasami? Czy potrafisz wyróżnić słowa mogące wpływać w znaczym stopniu na sentyment?
- jaka jest charakterystyka tekstu (np. długość, czystość)?


Opisz swoje ciekawe spostrzeżenia co do danych. Spróbuj na podstawie uzyskanych informacji zarekomendować dalsze działania w zakresie posiadanej wiedzy. 

Rezultatem pracy niech będzie prosty, krótki raport podsumowujący wiedzę, którą udało się uzyskać z EDA. Powinien zawierać sedno analiz i najważniejsze wnioski które udało się uzyskać. Raport powinien być w formacie Jupyter Notebook'a.

Co do formuły raportu - załóż, że pracujesz w dziale R&D renomowanej firmy i musisz przygotować krótkie streszczenie wiedzy dla swojego przełożonego, by mógł on podjąć decyzje o dalszych etapach prac. **Krótko, treściwie, same konkrety, wnioski i rekomendacje** :) 

### Zadanie 3 (1 pkt)

Porównywanie notebooków za pomocą `git` jest problematyczne, dlatego najlepiej przechowywać je z wyczyszczonym outputem a wykonane przechowywać za pomocą dvc w osobnej lokalizacji. Dodatkową zaletą jest redukcja rozmiaru repozytorium gdy notebook'i zawierają dużo obrazki.

Dodaj do DVC stage, który będzie wykonywał napisany wyczyszczony notebook zapisując w folderze `data` lub jego podfolderze, dodaj odpowiednie zależności. Na potrzeby laboratorium dodaj `cache: false` do wyjściowego notebooka, aby był trakowany przez git. Wykorzystaj komendę `jupyter nbconvert` lub narzędzie [papermill](https://papermill.readthedocs.io/en/latest/). 
