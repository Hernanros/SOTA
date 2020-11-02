November 2, 2020

<h2> Data Preprocessing: </h2>

[ ] - Remove Null Sentences <br>
[ ] - Deal with punctuation (either remove them or space them out ex: "where?" -> "where ? ") <br>
[ ] - Expanding Contractions ? <br>
[ ] - Removing digits ? <br>
[ ] - Removing Stop-Words ? <br>

<h2> Questions for Ivan and ourselves </h2>

[ ] - NLTK Stopwords include words like "not" which have important sentiment value - should we still remove them? <br>
[ ] - How should we deal with Contractions if we want to keep words like "not" (ex: "can't" -> "can t" or "can not") <br>
[ ] - For which metrics should we remove stop-words from, if at all? <br>
[ ] - How in general should we deal with punctuation? <br>

<h2> Results from EDA on Datasets </h2>

[ ] - We should remove latex examples. <br>
[ ] - In the QQP dataset have a lot of unicode (non-ascii) characters - what should we do with them? <br>
      We found code that maps non-ascii to the counterpart ascii character, but only works in a limited number of scenarios (weird letters but not on special characters or special punctuation) <br>


