# SyntacticSimilarity
Test Jaccard, Dice, TFIDF and BM25 documents similarities with Python Anaconda

In this example we use a corpus in French

If you  don't have Spacy you will need to install it first and also download a French Spacy Model needed for lemmatization.

conda install -c conda-forge spacy

python -m spacy download fr_core_news_md

We also use BM25 implementation by Koreyou https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8
