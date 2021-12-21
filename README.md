# SyntacticSimilarity
Test Jaccard, Dice, TFIDF and BM25 documents similarities with Python Anaconda

In this example we use a corpus in French in the file "CompaniesFrenchDesc.xlsx". But you can provide your own file with at least the 2 columns "Name" and "Description".

If you  don't have spaCy you will need to install it first and also download a French spaCy Model needed for lemmatization.

conda install -c conda-forge spacy

python -m spacy download fr_core_news_md

We also used BM25 implementation by Koreyou https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8


