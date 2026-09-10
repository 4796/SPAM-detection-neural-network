# Primena neuronskih mreža za predviđanje prevara u cyber bezbednosti (spam)

Projekat iz predmeta *Inteligentni sistemi*. Binarna klasifikacija SMS poruka (spam / ham) pomoću feedforward neuronske mreže (MLP) nad TF-IDF reprezentacijom teksta, uz sistematsku hiperparametarsku optimizaciju i analizu osetljivosti.

## Opis problema

Spam poruke predstavljaju jedan od najrasprostranjenijih oblika cyber prevare - koriste se za phishing, širenje malvera i finansijske prevare. Cilj ovog projekta je da se neuronska mreža istrenira da automatski razlikuje spam od legitimnih (ham) SMS poruka na osnovu sadržaja teksta, čime se demonstrira primena neuronskih mreža u zadatku detekcije prevare u cyber bezbednosti.

Zadatak je formulisan kao **binarna klasifikacija**: na osnovu teksta poruke, model predviđa verovatnoću da je poruka spam.

## Podaci

- **Izvor:** [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (Kaggle / UCI Machine Learning Repository)
- **Struktura:** 5572 SMS poruke na engleskom jeziku, svaka označena kao `ham` ili `spam`. Dataset je umereno neuravnotežen (~87% ham, ~13% spam).
- **Preprocesiranje:**
  1. Lowercase, uklanjanje URL-ova, interpunkcije i ne-slovnih karaktera
  2. Stratifikovan train/val/test split (70/15/15) - **pre** vektorizacije, kako bi se izbeglo curenje informacija (data leakage)
  3. TF-IDF vektorizacija (unigrami + bigrami, ograničen rečnik) - fitovana isključivo na trening skupu
  4. Neuravnoteženost klasa rešena kroz `pos_weight` u loss funkciji (weighted `BCEWithLogitsLoss`), umesto oversampling-a

Detaljna eksploratorna analiza (distribucija klasa, dužina poruka, najčešće reči po klasi) nalazi se u [`notebooks/spam_nn.ipynb`](notebooks/spam_nn.ipynb), sekcija 1.

## Arhitektura modela

`SpamMLP` ([`src/model.py`](src/model.py)) - konfigurabilan feedforward klasifikator:

- Ulaz: TF-IDF vektor poruke
- N skrivenih slojeva (`Linear → aktivacija → Dropout`), sa podesivim brojem slojeva, brojem neurona po sloju, tipom aktivacije (ReLU/Tanh) i dropout stopom
- Izlaz: 1 logit (verovatnoća spam klase se dobija primenom sigmoid funkcije)

Model je implementiran u PyTorch-u i omotan `skorch.NeuralNetBinaryClassifier` wrapperom kako bi bio kompatibilan sa scikit-learn API-jem (neophodno za `GridSearchCV`).

## Trening

Baseline model (razumne default vrednosti hiperparametara) treniran je da se potvrdi da ceo pipeline (podaci → model → optimizacija) radi ispravno pre pokretanja pune hiperparametarske pretrage. Prati se kretanje trening loss-a kroz epohe.

- Optimizator: Adam
- Loss: `BCEWithLogitsLoss` sa `pos_weight` za neuravnoteženost klasa
- Metrika za praćenje: F1-score (accuracy nije pouzdana zbog neuravnoteženosti klasa)

Baseline (2 skrivena sloja, 128 neurona, dropout 0.3, ReLU, 30 epoha) je nakon treninga postigao **F1 = 0.893** na validacionom skupu, čime je potvrđeno da pipeline radi ispravno pre hiperparametarske optimizacije. Grafik loss krive: `results/figures/baseline_loss.png` (detalji u [`notebooks/spam_nn.ipynb`](notebooks/spam_nn.ipynb), sekcija 3).

## Analiza osetljivosti modela i hiperparametarska optimizacija

Hiperparametarska pretraga izvedena je pomoću `GridSearchCV` (5-fold cross-validation, `scoring="f1"`) direktno nad PyTorch mrežom, preko skorch wrappera. Grid je namerno umeren (16 kombinacija × 5-fold = 80 treninga, 15 epoha po treningu) da pretraga ostane izvodljiva na CPU u razumnom vremenu. Batch size (64) i learning rate (1e-3) su fiksirani, a pretraženi su:

| Hiperparametar | Isprobane vrednosti |
|---|---|
| Broj skrivenih slojeva | 1, 2 |
| Broj neurona po sloju | 64, 128 |
| Dropout | 0.0, 0.3 |
| Aktivaciona funkcija | ReLU, Tanh |

**Najbolja kombinacija:** 1 skriveni sloj, 64 neurona, dropout 0.0, Tanh aktivacija, lr=1e-3 - CV F1 = 0.931.

Za svaki hiperparametar posebno analiziran je njegov uticaj na prosečan CV F1-score (marginalizovano preko ostalih kombinacija), kako bi se identifikovalo koji hiperparametri najviše utiču na performanse modela.

*(Rezultati pretrage, tabela najboljih kombinacija i grafici osetljivosti - vidi [`notebooks/spam_nn.ipynb`](notebooks/spam_nn.ipynb), sekcija 4, i `results/figures/hyperparameter_sensitivity.png`.)*

## Rezultati evaluacije

Finalni model (najbolji pronađeni hiperparametri, dotreniran na train+val) evaluiran je na potpuno odvojenom test skupu.

| Metrika | Vrednost |
|---|---|
| Accuracy | 0.986 |
| Precision | 0.955 |
| Recall | 0.938 |
| F1-score | 0.946 |
| ROC-AUC | 0.986 |

Confusion matrica, ROC i precision-recall krive: `results/figures/confusion_matrix.png`, `results/figures/roc_pr_curves.png`.

Model tačno prepoznaje 93.8% stvarnih spam poruka (recall), uz svega 4.5% lažno pozitivnih spam predikcija (1 − precision) - solidan balans za ovaj tip problema, gde su i propušteni spam (lažno negativni) i pogrešno blokirana legitimna poruka (lažno pozitivni) neželjeni.

## Diskusija

- **Uticaj hiperparametara:** najbolja kombinacija (1 skriveni sloj, 64 neurona, dropout 0.0, Tanh) sugeriše da za ovaj dataset (relativno mali, ~5500 uzoraka, 3000 TF-IDF feature-a) plići i jednostavniji model generalizuje bolje od dubljih/širih konfiguracija - očekivano, s obzirom na ograničenu količinu podataka u odnosu na broj parametara dubljih mreža (400k+ parametara kod 2 sloja × 128 neurona). Dropout=0.0 kao optimalna vrednost dodatno potvrđuje da model nije bio sklon overfitting-u čak i bez regularizacije, verovatno zbog kratkog treninga (15 epoha) i relativno niske dimenzionalnosti ulaza.
- **Ograničenja pristupa:** TF-IDF reprezentacija gubi redosled i kontekst reči (bag-of-words), pa fraze poput "not a scam" i "a scam" izgledaju slično modelu. Dataset je takođe relativno mali i sadrži samo engleske SMS poruke, što ograničava generalizaciju na druge jezike ili tipove poruka (npr. email spam).
- **Moguća unapređenja:** poređenje sa sekvencijalnom arhitekturom (LSTM/CNN nad word embeddings) bi moglo dodatno poboljšati rezultate hvatanjem konteksta; širi hiperparametarski grid (uključujući learning rate i batch size) uz GPU ubrzanje bi omogućio temeljitiju pretragu.

## Zaključak

Feedforward neuronska mreža nad TF-IDF vektorima pokazala se kao efikasan i tačan pristup za detekciju spam poruka (F1 = 0.946, ROC-AUC = 0.986 na test skupu), uprkos relativnoj jednostavnosti arhitekture. Rezultati potvrđuju da je primena neuronskih mreža opravdana za ovaj zadatak detekcije cyber prevare - čak i minimalna mreža (1 skriveni sloj) postiže visoku tačnost kada je ulazna reprezentacija (TF-IDF) dovoljno informativna, što ukazuje da za jednostavnije tekstualne zadatke poput ovog kompleksnost modela nije glavni ograničavajući faktor.

## Struktura repozitorijuma

```
├── data/raw/spam.csv          # sirovi dataset
├── src/
│   ├── data_utils.py          # učitavanje, čišćenje, split, TF-IDF vektorizacija
│   └── model.py                # SpamMLP (PyTorch)
├── notebooks/spam_nn.ipynb    # kompletan pipeline: podaci → arhitektura → trening → HP optimizacija → evaluacija
├── results/                    # sačuvani grafici i finalni model
├── requirements.txt
└── LICENSE
```

## Pokretanje

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/spam_nn.ipynb
```

Dataset preuzeti sa [Kaggle-a](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) i sačuvati kao `data/raw/spam.csv`.

## Licenca

Ovaj projekat je dostupan pod [MIT licencom](LICENSE).
