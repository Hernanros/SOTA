November 2, 2020

<h2> Data Preprocessing: </h2>

[ ] - Remove Null Sentences
[ ] - Deal with punctuation (either remove them or space them out ex: "where?" -> "where ? ")
[ ] - Expanding Contractions ?
[ ] - Removing digits ?
[ ] - Removing Stop-Words ?

<h2> Questions for Ivan and ourselves </h2>

[ ] - NLTK Stopwords include words like "not" which have important sentiment value - should we still remove them?
[ ] - How should we deal with Contractions if we want to keep words like "not" (ex: "can't" -> "can t" or "can not")
[ ] - For which metrics should we remove stop-words from, if at all?
[ ] - How in general should we deal with punctuation? 

<h2> Results from EDA on Datasets </h2>

[ ] - We should remove latex examples.
[ ] - In the QQP dataset have a lot of unicode (non-ascii) characters - what should we do with them?
      We found code that maps non-ascii to the counterpart ascii character, but only works in a limited number of scenarios (weird letters but not on special characters or special punctuation)


