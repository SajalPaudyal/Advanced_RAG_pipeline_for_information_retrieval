# Advanced Retrieval Augmented Generation (RAG) project for a movie recommendation system.

This project is a modification (or an optimization) for the previous project of a [naïve RAG system](https://github.com/SajalPaudyal/Implementing_basic_RAG_for_movie_recommendation).

The Advanced RAG modifies various components from the naïve RAG system and tries to optimize the RAG paradigm. There are various approaches in modifying the traditional RAG and generating answers that are accurate,
factual, and contextual. We use the following methods to modify the naïve RAG:

**Step 1: Query Transformation**:
The query is modified into a much descriptive query by simply adding a related term to the original query. This asks the LLM to add a *relevant term* with the query so that the context is much clearer and it facilitates the generation.

**Step 2: Query Routing**:
Instead of just using the query to search the vector database, we want to conduct a different search and control flow to the system. In our case we use the *keyword router* and *semantic router*
    - *Keyword router*: It is a set of logical rules with if/else scenario where routes are selected by matching keyword between query and set of options.
    - *Semantic router*: It is used to decide the best route. We have a list of examples and the associated routes which are embedded and saved in a vector database. When our system receives a new query, if we find a specific keyword in the query then the search is textual else it is a vector.

**Step 3: Hybrid Search Method**:
*Keyword Based* search is a search based on exact matching of a certain word, while *Vector or Semantic search* finds the semantic meaning of a query but the ***Hybrid Search*** takes the best of two worlds and combines a model for a vector
as well as keyword search.

**Step 4: Re-ranking**:
After routing and searching the keywords, we cannot returned all the retrieved chunks both, because they will not fit in the context window of the model, and also the LLM might overfit with the overwhelming information. Hence, for this we need to
maximize the document retrieved and also at the same time maximize the LLM-recall. This strategy is called **Re-Ranking**.
    *Re-ranking step 1*: We conduct a classic retrieval and collect a large number of relevant chunks.
    *Re-ranking step 2:* We use a re-ranker (in our case a pre-trained BERT model) to reorder the chunks and then select the top-k (k = 5) chunks that are provided to the LLM.

**Step 5: Summarization**: Context can also contain information that are redundant. LLMs are sensitive to these potential noises and hence reducing them would help in better generation. In our case, we use [summarization](https://huggingface.co/docs/transformers/en/tasks/summarization) module provided by transformers and set the limit to avoid losing too much information. 

Once all these steps are followed, they are assembeled into one single pipeline and executed.


----

## Getting Started
----
- Python 3.14 (used and recommended)
- pip and virtualenv (or conda)
- Jupyter Notebooks or JupyterLab

----
### Setup (using venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Notebook Overview
----
1. Advanced RAG movie recommendation pipeline (`advanced_rag_for_movie_recommendation.ipynb`)
    - Implements the Advanced Retrieval Augmented Generation (RAG) pipeline as described above. 
    - This project uses various APIs of which you will find the link exactly before the usage block.

----
## Relevant research papers:
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172)
- [Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG](https://arxiv.org/pdf/2409.07691)
- [Adapting Language Models to Compress Contexts](https://arxiv.org/pdf/2305.14788)
- [A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2405.06211)