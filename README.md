# Reddit Post Classification
Classifying political posts on reddit.

#### Steps:

* Mine data using PushShift and Reddit API.
* Extract keywords using TextRank
* Build a simple logistic regression on relative word frequencies.

#### Motivation and Use Case

Annoyed by US centric political post on reddit. A small final word frequency table (~2 MB) and great accuracy of a simple logistic regression implying that it can be used to classify and hide posts on user's end itself without using any significant amount of resources.

Final dataset can be found [here.](https://www.kaggle.com/anjay23/word-frequency-in-political-and-nonpol-subreddit)

#### TODO

Make a browser plugin.
