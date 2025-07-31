# `LIPE`
- The latent interview protocols engineer (LIPE) is a Python utility for mapping out the underlying structure of a corpus of interviews.
	- The underlying assumption is that in interviews on a certian topic, there will be an underlying set of key questions that are explored. These questions often follow a certain logic (e.g., if interviewer says X then ask Y else ask Z). We refer to this underlying set of key questions and how they connect as the *interview protocol*. In more structured interviews, this protocol might be known before hand, but often, interviewers may diverge from it as the interview question molds to the nuances of the phenomena. Other times, there may not even be an explicit protocol to start with. The goal of LIPE is to recover this interview protocol as it is latently expressed across a corpus of interviews.
   - Protocol questions are distinct from incidental questions, such as specific follow up questions or clarification questions, in that protocol questions characterize the entire corpus of interviews. As such, we can expect protocol questions to occur in a significant portion of the interviews, even if they are worded slightly differently.
   - Using sentence embedding models, we can represent interviewer lines in a high dimensional space. We can then take significantly large clusters of these embeddings to represent protocol questions.
   - Finally, we can examine the sequences of protocol questions in each interview to build a map of how interviews in the corpus typically proceed.
- Qualitative input
   - LIPE provides an interface for researchers to qualitatively examine, label, merge, and split labels.
- Navigating mapped out interview corpora
   - Once a corpus of interviews has been mapped out, researchers can use it to navigate the otherwise unwieldy corpus.
   - LIPE allows researchers to extract responses to particular protocol questions across the corpus. These responses include responses to any follow up questions.
- LIPE takes as an input a structured dataframe where each row is a line from an interview. Each line needs to have an interview ID, line ID, and speaker ID in addition to the text for that line.
   - A utility to automatically identify interviewer lines is currently under development.

## Process
1. Load labeled interview lines. LIPE will automatically start the clustering process.
2. Examine identified protocol question clusters.
   a. Use visualization tools to examine clusters systematically.
   b. Circle back to step 1 if clustering hyperparameters need to be adjusted.
   c. Otherwise, merge/split/exclude/label clusters as necessary.
3. Visualize interview protocol as a directed graph.

## TODO
- Saving/loading functionality
- ~~Interview lines export with subsetting~~
- Include more outlier reduction strategies
- ~~Manual relabeling~~
- ~~Topic merging/splitting. It's still very buggy and BERTopic doesn't make it easy to modify fitted models.~~
- Question-level EDA
- Fit new interviewer lines
- Example/vignette