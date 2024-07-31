# Project Title

Connecting the Dots: Enhanced Data Discovery and Building Trust
## Project Overview

Discover relationships within datasets hosted on data.gov and also build an advanced RAG chatbot for data discoverability

### Table of Contents

- [Data Collection and Processing](#data-collection-and-processing)
- [Embedding Projector Creation](#embedding-projector)
- [RAG Chatbot](#rag-chatbot)
- [Querying and Response Generation](#querying-and-response-generation)
- [Summary](#summary)

## Data Collection and Processing

We used the Data.Gov api to collect 10000 datasets and hosted our own MilvusDB locally to chunk and embed the data. We embedded the data using the arctic-embed-m model. We used advanced techniques such as Recursive Character Splitting for creating chunks.

## Embedding Projector
Introducing an advanced embedding projector designed to uncover and visualize intricate relationships across different datasets. This innovative tool transforms complex data into clear, interactive visualizations, enabling users to intuitively explore, analyze, and derive meaningful insights. Perfect for data scientists, researchers, and analysts, our embedding projector simplifies the discovery process, making it easier to identify patterns, trends, and connections that drive informed decision-making.

## RAG

Introducing our cutting-edge RAG Chatbot, meticulously built from scratch to revolutionize data discoverability on our website. Leveraging advanced techniques such as query decomposition and hybrid search, our chatbot excels in understanding and responding to complex queries with remarkable accuracy. Query decomposition breaks down intricate questions into manageable components, ensuring precise and relevant answers. Hybrid search combines the best of both keyword-based and semantic search methods, offering users comprehensive and contextually rich responses. This powerful tool not only enhances user engagement but also drives efficiency and satisfaction by providing swift, intelligent, and context-aware interactions, making data exploration seamless and intuitive.

## Trustworthy AI

Our RAG Chatbot, combined with the embedding projector, is a powerful tool for verifying misinformation, providing accurate statistics and visual insights. It ensures data integrity by cross-referencing and visualizing relationships within datasets for reliable, informed decision-making.

## Summary

Our project encompasses the development of two innovative tools: an advanced embedding projector and a cutting-edge RAG Chatbot, both meticulously built from scratch. The embedding projector reveals intricate relationships within datasets through clear, interactive visualizations, enhancing data exploration and analysis. The RAG Chatbot leverages query decomposition and hybrid search techniques to provide accurate, contextually rich responses, significantly improving data discoverability on our website. Together, these tools also serve as powerful allies in verifying misinformation, offering reliable statistics and insights for informed decision-making. This comprehensive project redefines user engagement, efficiency, and satisfaction in data interaction and validation.


## Getting Started

### Dependencies

### Installing

```bash
pip install -r requirements.txt
streamlit run main.py
cd chatbot
uvicorn main:app --reload
```



### URLs (Hosted application w/o chat):

* https://gsa-hackathon-binoloop-dataset-convo.streamlit.app/



