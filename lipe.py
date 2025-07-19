import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from scipy.cluster import hierarchy as sch
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import random


def e5_preprocessor(s):
    return "query: " + s


def linkage_function(x):
    return sch.linkage(x, 'single', optimal_ordering=True)

# BERTopic monkey-patch for topic removal


def remove_topic(self, topic_id: int):
    # 1. Safety checks
    if self.topic_representations_ is None:
        raise ValueError("Model not yet fitted.")
    if topic_id not in self.topic_representations_:
        raise KeyError(f"Topic {topic_id} not found.")

    # 2. Find the row index in matrices/arrays
    #    Assumes keys sorted in ascending order match rows in c_tf_idf_ & topic_embeddings_
    sorted_ids = sorted(self.topic_representations_.keys())
    idx = sorted_ids.index(topic_id)

    # 3. Remove from c-TF-IDF matrix
    if self.c_tf_idf_ is not None:
        mask = np.arange(self.c_tf_idf_.shape[0]) != idx
        self.c_tf_idf_ = self.c_tf_idf_[mask]

    # 4. Remove from topic embeddings
    if self.topic_embeddings_ is not None:
        self.topic_embeddings_ = np.delete(self.topic_embeddings_, idx, axis=0)

    # 5. Update topic sizes
    self.topic_sizes_.pop(topic_id, 0)

    # 6. Reassign documents → outliers (–1)
    if hasattr(self, "topics_") and self.topics_ is not None:
        self.topics_ = [t if t != topic_id else -1 for t in self.topics_]
        # Update outlier count
        self.topic_sizes_[-1] = sum(t == -1 for t in self.topics_)

    # 7. Adjust probabilities
    if getattr(self, "probabilities_", None) is not None:
        probs = self.probabilities_
        if probs.ndim == 2:
            self.probabilities_ = np.delete(probs, idx, axis=1)
        else:
            # 1D probabilities → zero out removed-topic docs
            self.probabilities_ = [
                p if topic != topic_id else 0 for p, topic in zip(probs, self.topics_)
            ]

    # 8. Clean out representations & labels
    self.topic_representations_.pop(topic_id, None)
    self.representative_docs_.pop(topic_id, None)
    if self.representative_images_:
        self.representative_images_.pop(topic_id, None)
    self.topic_aspects_.pop(topic_id, None)
    if getattr(self, "_topic_id_to_zeroshot_topic_idx", None) is not None:
        self._topic_id_to_zeroshot_topic_idx.pop(topic_id, None)

    # 9. Custom labels
    if isinstance(self.custom_labels_, list):
        # Align by sorted_ids
        self.custom_labels_ = [
            lbl for id_, lbl in zip(sorted_ids, self.custom_labels_)
            if id_ != topic_id
        ]

    # 10. Update the TopicMapper
    if hasattr(self, "topic_mapper_") and self.topic_mapper_:
        mappings = self.topic_mapper_.get_mappings()
        new_maps = {
            k: v for k, v in mappings.items()
            if k != topic_id and v != topic_id
        }
        self.topic_mapper_.mappings = new_maps


BERTopic.remove_topic = remove_topic


class LIPE:
    def __init__(self, data: pd.DataFrame,
                 speaker_col='speaker_id',
                 text_col='text',
                 interview_col='interview_id',
                 line_col='line_id',
                 tm_preprocessor=None,
                 interviewer_id=0,
                 model=None,
                 min_topic_size=None,
                 random_state=42):

        self.data = data.copy()
        self.speaker_col = speaker_col
        self.text_col = text_col
        self.interview_col = interview_col
        self.line_col = line_col
        self.interviewer_id = interviewer_id

        # Count unique interviews
        self.n_interviews = self.data[interview_col].nunique()
        if self.n_interviews < 10:
            warnings.warn(
                f"Only {self.n_interviews} unique interviews detected. LIPE may not be able to "
                "distinguish shared protocol questions from interview-specific follow-ups. "
                "At least 10 interviews are recommended.", UserWarning
            )

        # Determine default min_topic_size only if we're creating a model
        self.min_topic_size = max(self.n_interviews // 4, 5) if min_topic_size is None else min_topic_size
        self.embedding_model_name = None

        if model is None:
            self.embedding_model_name = 'intfloat/e5-base-v2'
            if tm_preprocessor is None:
                tm_preprocessor = e5_preprocessor
            model = self._load_topic_model(random_state=random_state)
        elif isinstance(model, str):
            self.embedding_model_name = model
            model = self._load_topic_model(random_state=random_state)
        elif isinstance(model, BERTopic):
            pass  # user-provided, use as-is
        else:
            raise ValueError("`model` must be None, a string, or a BERTopic instance.")

        self.model = model

        # Prepare interviewer lines
        self.interviewer_lines = self.data[self.data[speaker_col] == interviewer_id].reset_index(drop=True)

        if tm_preprocessor:
            self.interviewer_lines_preprocessed = self.interviewer_lines[self.text_col].apply(tm_preprocessor).to_list()
        else:
            self.interviewer_lines_preprocessed = self.interviewer_lines[self.text_col].to_list()

        self.topic_labels, self.topic_probs = self.model.fit_transform(self.interviewer_lines_preprocessed)
        self.topic_info = self.model.get_topic_info()
        self.questions = self._build_questions()
        self.interviewer_lines['topic'] = self.topic_labels

    def _load_topic_model(self, random_state=42):
        embed_model = SentenceTransformer(self.embedding_model_name)
        umap_model = UMAP(n_neighbors=self.min_topic_size,
                          n_components=5,
                          min_dist=0.0,
                          metric='cosine',
                          random_state=random_state)
        hdbscan_model = HDBSCAN(min_cluster_size=self.min_topic_size,
                                min_samples=self.min_topic_size // 2,
                                metric='euclidean',
                                cluster_selection_method='eom',
                                prediction_data=True)
        return BERTopic(embedding_model=embed_model,
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        min_topic_size=self.min_topic_size)

    def _build_questions(self):
        questions = {}
        for topic_id in self.topic_info['Topic'].unique():
            questions[topic_id] = ProtocolQuestion(topic_id, self)
        return questions

    def detailed_examples(self, topics=None, n=5, example_type='mixed', show_unprocessed=True):
        """
        Return example interviewer docs from selected topic(s).

        Parameters:
        - topics: int or list of ints. If None, includes all topics.
        - n: number of examples per topic
        - example_type: one of ['representative', 'random', 'mixed']

        Returns:
        - DataFrame with topic_id, count, words, and example docs
        """
        if isinstance(topics, int):
            topics = [topics]
        elif topics is None:
            topics = self.topic_info['Topic'].unique().tolist()

        info = []
        all_docs = self.model.get_document_info(self.interviewer_lines_preprocessed)
        if show_unprocessed:
            all_docs.Document = self.interviewer_lines.text.to_list()

        for topic_id in topics:
            if topic_id == -1 or topic_id not in self.topic_info['Topic'].values:
                continue

            docs = all_docs[all_docs.Topic == topic_id]
            count = len(docs)
            words = [w[0] for w in self.model.get_topic(topic_id)]

            if example_type == 'representative':
                examples = docs[docs.Representative_document]
                if len(examples) > n:
                    examples = examples.sample(n)
            elif example_type == 'random':
                examples = docs
                if len(examples) > n:
                    examples = examples.sample(n)
            elif example_type == 'mixed':
                reps = docs[docs.Representative_document]
                if len(reps) > n / 2:
                    reps = reps.sample(n // 2)

                bal = n - len(reps)
                others = docs[~docs.Representative_document]
                if len(others) > bal:
                    others = others.sample(bal)

                examples = pd.concat([reps, others])
            else:
                raise ValueError("example_type must be one of ['representative', 'random', 'mixed']")

            info.append({
                'topic_id': topic_id,
                'count': count,
                'words': words,
                'examples': list(zip(examples['Probability'], examples['Document']))
            })

        return pd.DataFrame(info)

    def visualize_topics(self):
        return self.model.visualize_topics()

    def visualize_documents(self):
        return self.model.visualize_documents(self.interviewer_lines[self.text_col].to_list())

    def visualize_hierarchy(self, linkage_function=linkage_function):
        if not hasattr(self, 'hierarchical_topics') or self.hierarchical_topics is None:
            self.hierarchical_topics = self.model.hierarchical_topics(
                self.interviewer_lines[self.text_col].tolist(),
                linkage_function=linkage_function
            )
        return self.model.visualize_hierarchy(hierarchical_topics=self.hierarchical_topics)

    def merge_topics(self, topic_ids: list, new_topic_id=None):
        self.model.merge_topics(self.interviewer_lines[self.text_col].tolist(), topic_ids, new_topic=new_topic_id)
        self._reassign_topics()

    def split_topic(self, topic_id: int, nr_topics=None):
        self.model.reduce_outliers(self.interviewer_lines[self.text_col].tolist(), strategy='both', threshold=0.5)
        self.model.hierarchical_topics(self.interviewer_lines[self.text_col].tolist())
        self._reassign_topics()

    def _reassign_topics(self):
        self.topic_labels, self.topic_probs = self.model.transform(self.interviewer_lines_preprocessed)
        self.model.topics_ = self.topic_labels
        if hasattr(self.model, 'probabilities_') and self.topic_probs is not None:
            self.model.probabilities_ = self.topic_probs
        self.model.topic_sizes_ = pd.Series(self.topic_labels).value_counts().to_dict()

        self.topic_info = self.model.get_topic_info()
        self.questions = self._build_questions()
        self.interviewer_lines['topic'] = self.topic_labels

    def label_full_data(self):
        self.data['topic'] = -4
        interviewer_mask = self.data[self.speaker_col] == self.interviewer_id
        self.data.loc[interviewer_mask, 'topic'] = self.topic_labels
        return self.data

    def ignored_topics(self):
        return {i for i in self.questions if self.questions[i].ignore}

    def get_transitions(self, ignore_topics={-1}, ignore_self_transitions=True):
        if ignore_topics is None:
            ignore_topics = set()
        else:
            ignore_topics = set(ignore_topics)

        ignore_topics = self.ignored_topics().union(ignore_topics)

        df = self.interviewer_lines[[self.interview_col, self.line_col, 'topic']].sort_values(
            [self.interview_col, self.line_col]
        )

        transitions = []

        for interview_id, group in df.groupby(self.interview_col):
            topics = group['topic'].tolist()

            # Insert synthetic start and end nodes
            topics = [-3] + topics + [-2]
            topics = [topic for topic in topics if topic not in ignore_topics]

            for i in range(len(topics) - 1):
                current, next_ = topics[i], topics[i + 1]
                transitions.append((current, next_))

        transitions_df = pd.DataFrame(transitions, columns=['topic', 'next_topic'])

        if ignore_self_transitions:
            transitions_df = transitions_df[transitions_df["topic"] != transitions_df["next_topic"]]

        return transitions_df

    def tabulate_transitions(self, ignore_topics={-1}):
        transitions = self.get_transitions(ignore_topics=ignore_topics)
        return transitions.value_counts().reset_index(name='count')

    def build_graph(self, ignore_topics={-1}):
        edges = self.tabulate_transitions(ignore_topics=ignore_topics)
        G = nx.DiGraph()
        for _, row in edges.iterrows():
            G.add_edge(row['topic'], row['next_topic'], weight=row['count'])
        self.graph = G

    def plot_graph(self,
                   min_count=0,
                   seed=42,
                   spring_k=None,
                   iterations=50,
                   pos=None,
                   title="Latent Interview Protocol Graph",
                   topic_labels={-3: 'START', -2: 'END'},
                   layout_adjustments=None,
                   figsize=(10, 10),
                   default_node_color='skyblue',
                   node_color_map={-3: 'lightgray', -2: 'lightgray'}):
        """
        Return a directed graph of the inferred interview structure.

        This function uses NetworkX's implementation of the Fruchterman-Reingold force-directed
        algorithm. Some of that function's parameters like seed and k can be passed, but for
        more customized plotting use self.graph to construct a graph from scratch.

        Parameters:
        - min_count: only draw edges with weight >= min_count
        - seed: seed for layout reproducibility
        - spring_k: repulsion parameter for spring_layout
        - pos: manually provided position dictionary (optional)
        - title: title for the graph
        - topic_labels: dict mapping topic_id to label (e.g., from BERTopic.get_topic_info)
        - layout_adjustments: dict of manual coordinate adjustments: {node: (dx, dy)}
        - figsize: tuple for figure size
        - node_color_map: dict {node_id: color} to customize node colors

        Returns:
        - None (renders plot)
        """

        if not hasattr(self, 'graph') or self.graph is None:
            self.build_graph()

        # Filter edges
        filtered_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True) if d["weight"] >= min_count]
        G = nx.DiGraph()
        G.add_edges_from(filtered_edges)

        # Base layout
        if pos is None:
            pos = nx.spring_layout(G, seed=seed, k=spring_k, iterations=iterations)

        # Apply manual layout nudges
        if layout_adjustments:
            for node, (dx, dy) in layout_adjustments.items():
                if node in pos:
                    pos[node] = pos[node] + np.array([dx, dy])

        # Node weights (based on total edge traffic)
        node_weights = {}
        for u, v, d in G.edges(data=True):
            node_weights[u] = node_weights.get(u, 0) + d['weight']
            node_weights[v] = node_weights.get(v, 0) + d['weight']

        max_w = max(node_weights.values()) if node_weights else 1
        node_sizes = {n: 300 + 1000 * (node_weights.get(n, 0) / max_w) for n in G.nodes()}

        # Node colors
        node_colors = [node_color_map.get(n, default_node_color) for n in G.nodes()]

        # Draw nodes
        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(
            G, pos,
            node_size=[node_sizes[n] for n in G.nodes()],
            node_color=node_colors
        )

        # Labels
        node_labels = {}
        for node in G.nodes:
            question = self.questions.get(node, None)
            node_labels[node] = question.label if question else 'NO LABEL'
        if topic_labels:
            node_labels.update(topic_labels)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

        # Edge weights and widths
        weights = [d['weight'] for (_, _, d) in filtered_edges]
        max_weight = max(weights) if weights else 1
        line_widths = [1 + 5 * (w / max_weight) for w in weights]

        nx.draw_networkx_edges(
            G, pos,
            edgelist=filtered_edges,
            arrowstyle='->',
            arrowsize=20,
            width=line_widths,
            edge_color='gray',
            connectionstyle='arc3,rad=0.05'
        )

        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def get_answers(self, question_id, include_question=True, merge_transcript_lines=True):
        ignored_topics = self.ignored_topics().union({-4, -1})
        df = self.label_full_data()
        df.sort_values(by=[self.interview_col, self.line_col])
        keep = []
        capturing = False
        for index, row in df.iterrows():
            if capturing:
                if row['topic'] not in ignored_topics and row['topic'] != question_id:
                    capturing = False
                else:
                    if row[self.speaker_col] == 0 and not include_question:
                        continue
                    keep.append(index)
            else:
                if row['topic'] == question_id:
                    capturing = True
                    if include_question:
                        keep.append(index)

        answers = df[df.index.isin(keep)]

        if not include_question and merge_transcript_lines:
            answers = answers.groupby(self.interview_col, as_index=False)[self.text_col].agg(' '.join)

        return answers


class ProtocolQuestion:
    def __init__(self, topic_id, lipe):
        self.topic_id = topic_id
        self.lipe = lipe  # back-reference to parent
        self._info = self.lipe.topic_info[self.lipe.topic_info.Topic == topic_id].iloc[0]
        self.label = self._info['Name']
        self.ignore = False

    def get_examples(self, n=5, example_type="mixed", show_unprocessed=True):
        return self.lipe.detailed_examples(topics=[self.topic_id],
                                           n=n,
                                           example_type=example_type,
                                           show_unprocessed=show_unprocessed)['examples'].iloc[0]

    def get_answers(self, include_question=True, merge_transcript_lines=True):
        return self.lipe.get_answers(self.topic_id, include_question=include_question,
                                     merge_transcript_lines=merge_transcript_lines)

    @property
    def representative_words(self):
        return self._info['Representation']

    @property
    def count(self):
        return self._info['Count']

    @property
    def has_split(self):
        return hasattr(self, '_split_state')

    @property
    def split_preview(self):
        if not self.has_split:
            return None
        return self._split_state['model'].get_topic_info()

    def split(self, min_topic_size, custom_model=None, embedding_model_name=None, random_state=42):
        # Extract relevant interviewer lines
        mask = self.lipe.interviewer_lines['topic'] == self.topic_id
        texts = self.lipe.interviewer_lines.loc[mask, self.lipe.text_col].tolist()

        if not texts:
            raise ValueError(f"No documents found for topic {self.topic_id}.")

        # Use user-provided or new model
        if custom_model:
            model = custom_model
        else:
            embed_model = SentenceTransformer(embedding_model_name or self.lipe.embedding_model_name)
            umap_model = UMAP(n_neighbors=min_topic_size, n_components=5, min_dist=0.0,
                              metric='cosine', random_state=random_state)
            hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, min_samples=max(1, min_topic_size // 2))
            model = BERTopic(embedding_model=embed_model,
                             umap_model=umap_model,
                             hdbscan_model=hdbscan_model,
                             min_topic_size=min_topic_size)

        new_topics, _ = model.fit_transform(texts)

        self._split_state = {
            'model': model,
            'new_topics': new_topics,
            'docs': texts
        }
        return self._split_state['model'].get_topic_info()

    def get_split_examples(self, topic, n=10):
        docs = [d for t, d in zip(self._split_state['new_topics'], self._split_state['docs']) if t == topic]
        if len(docs) > n:
            docs = random.sample(docs, n)
        return docs

# WIP
#    def reduce_split_outliers(self, strategy='distributions', final=False):
#        new_topics = topic_model.reduce_outliers(self._split_state['docs'], self._split_state['new_topics'], strategy=strategy)

    def commit_split(self, min_similarity=0.9):
        """
        Merge the split subtopics into the main LIPE model using BERTopic's merge_models.
        """
        if not hasattr(self, '_split_state'):
            raise RuntimeError("No split has been run. Call `.split()` first.")

        split_model = self._split_state['model']

        # Remove the topic from the main model
        self.lipe.model.remove_topic(self.topic_id)
        del self.lipe.questions[self.topic_id]

        # Merge the split model into the main model
        self.lipe.model = BERTopic.merge_models(
            [self.lipe.model, split_model],
            min_similarity=min_similarity
        )

        # Reassign topics in the LIPE model
        self.lipe._reassign_topics()

        # Clean up the split state
        del self._split_state
