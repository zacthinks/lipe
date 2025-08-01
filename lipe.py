import os
import re
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster import hierarchy as sch

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

import spacy
from spacy.pipeline import Sentencizer
from spacy.util import is_package
import spacy.cli


def e5_preprocessor(s):
    return "query: " + s


def linkage_function(x):
    return sch.linkage(x, 'single', optimal_ordering=True)


def slugify(text):
    return re.sub(r'\W+', '_', text.strip()).strip('_')


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
                 random_state=42,
                 min_prob=.5):

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
        self.interviewer_lines = self.data[self.data[speaker_col] == interviewer_id]

        if tm_preprocessor:
            self.interviewer_lines_preprocessed = self.interviewer_lines[self.text_col].apply(tm_preprocessor).to_list()
        else:
            self.interviewer_lines_preprocessed = self.interviewer_lines[self.text_col].to_list()

        topic_labels, topic_probs = self.model.fit_transform(self.interviewer_lines_preprocessed)
        topic_labels = [t if p >= min_prob else -1 for t, p in zip(topic_labels, topic_probs)]
        self.topic_info = self.model.get_topic_info()
        self.topic_info['Parent_Topic'] = np.nan
        self.topic_info['Parent_Topic'] = self.topic_info['Parent_Topic'].astype('Int64')
        self.questions = self._build_questions()
        self.question_merges = {}
        self.interviewer_lines['topic'] = topic_labels
        self.interviewer_lines['topic_prob'] = topic_probs
        counts = Counter(topic_labels)
        self.topic_info['Count'] = self.topic_info['Topic'].map(counts)

    @property
    def topic_map(self):
        return {q.topic_id: q.root_topic for q in self.questions.values()}

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

    def load_spacy(self, model="en_core_web_sm", disable=("parser", "ner")):
        try:
            self.spacy_model = spacy.load(model, disable=disable)
        except OSError:
            if not is_package(model):
                raise ValueError(f"spaCy model '{model}' not found and not downloadable.")
            try:
                spacy.cli.download(model)
                self.spacy_model = spacy.load(model, disable=disable)
            except Exception as e:
                raise ValueError(f"Failed to download and load spaCy model '{model}': {e}")

    def next_question_index(self):
        return max(self.questions.keys()) + 1

    def _build_questions(self):
        questions = {}
        for topic_id in self.topic_info['Topic'].unique():
            questions[topic_id] = ProtocolQuestion(topic_id, self)
        return questions

    def detailed_examples(self, topics=None, n=5, show_processed=False):
        """
        Return example interviewer docs from selected topic(s).

        Parameters:
        - topics: int or list of ints. If None, includes all topics.
        - n: number of examples per topic
        - show_processed: some encoding models require processing of texts--if true, show these processed versions, if not show raw texts

        Returns:
        - DataFrame with topic_id, count, words, and example docs
        """
        if isinstance(topics, int):
            topics = [topics]
        elif topics is None:
            topics = self.topic_info['Topic'].unique().tolist()

        info = []
        all_docs = self.interviewer_lines.copy()
        if show_processed:
            all_docs[self.text_col] = self.interviewer_lines_preprocessed

        for topic_id in topics:
            if topic_id not in self.topic_info['Topic'].values:
                continue

            all_descendants = self.questions[topic_id].merged_descendants

            docs = all_docs[all_docs['topic'].isin(all_descendants)]
            count = len(docs)
            words = self.topic_info[self.topic_info['Topic'] == -1]['Representation'].iloc[0]

            if len(docs) > n:
                docs = docs.sample(n)

            info.append({
                'topic_id': topic_id,
                'count': count,
                'words': words,
                'examples': list(zip(docs['topic_prob'], docs[self.text_col]))
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

    def label_full_data(self):
        self.data['topic'] = -4
        interviewer_mask = self.data[self.speaker_col] == self.interviewer_id
        self.data.loc[interviewer_mask, 'topic'] = self.interviewer_lines['topic'].to_numpy()
        return self.data

    def ignored_topics(self, include_merge_outliers=False):
        ignored = {i for i in self.questions if self.questions[i].ignore}
        if include_merge_outliers:
            ignored = {i for i in ignored if self.questions[i].has_children}
        return ignored

    def get_transitions(self, ignore_topics={-1}, ignore_self_transitions=True, ignore_merge_outliers=False):
        if ignore_topics is None:
            ignore_topics = set()
        else:
            ignore_topics = set(ignore_topics)

        ignore_topics = self.ignored_topics(ignore_merge_outliers).union(ignore_topics)

        df = self.interviewer_lines[[self.interview_col, self.line_col, 'topic']].sort_values(
            [self.interview_col, self.line_col]).copy()
        df['topic'] = df['topic'].map(self.topic_map)

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

    def tabulate_transitions(self, ignore_topics={-1}, ignore_self_transitions=True, ignore_merge_outliers=False):
        transitions = self.get_transitions(ignore_topics=ignore_topics,
                                           ignore_self_transitions=ignore_self_transitions,
                                           ignore_merge_outliers=ignore_merge_outliers)
        return transitions.value_counts().reset_index(name='count')

    def build_graph(self, ignore_topics={-1}, ignore_self_transitions=True, ignore_merge_outliers=False):
        edges = self.tabulate_transitions(ignore_topics=ignore_topics,
                                          ignore_self_transitions=ignore_self_transitions,
                                          ignore_merge_outliers=ignore_merge_outliers)
        G = nx.DiGraph()
        for _, row in edges.iterrows():
            G.add_edge(row['topic'], row['next_topic'], weight=row['count'])
        self.graph = G

    def plot_graph(self,
                   min_count=0,
                   seed=42,
                   spring_k=3,
                   iterations=200,
                   pos={-3: (-1, -1), -2: (1, 1)},
                   title="Latent Interview Protocol Graph",
                   topic_labels={-3: 'START', -2: 'END'},
                   label_ids=False,
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
        else:
            print("Plotting previously built graph--rerun self.build_graph() to modify the graph.")

        # Filter edges
        filtered_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True) if d["weight"] >= min_count]
        G = nx.DiGraph()
        G.add_edges_from(filtered_edges)

        # Base layout
        pos = nx.spring_layout(G, seed=seed, k=spring_k, iterations=iterations, pos=pos)

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
        if label_ids:
            node_labels = {k: f"{str(k)}: {v}" for k, v in node_labels.items()}
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
        family_topics = self.questions[question_id].merged_descendants
        df = self.label_full_data()
        df.sort_values(by=[self.interview_col, self.line_col])
        keep = []
        capturing = False
        for index, row in df.iterrows():
            if capturing:
                if row['topic'] not in ignored_topics and row['topic'] not in family_topics:
                    capturing = False
                else:
                    if row[self.speaker_col] == 0 and not include_question:
                        continue
                    keep.append(index)
            else:
                if row['topic'] in family_topics:
                    capturing = True
                    if include_question:
                        keep.append(index)

        answers = df[df.index.isin(keep)]

        if not include_question and merge_transcript_lines:
            answers = answers.groupby(self.interview_col, as_index=False)[self.text_col].agg(' '.join)

        return answers

    def export_answers(self, folder, use_graph_topics=True, topics=None, transcript_labels=None):
        os.makedirs(folder, exist_ok=True)
        if use_graph_topics:
            topic_ids = list(self.graph.nodes)
        else:
            topic_ids = topics if topics is not None else list(self.questions.keys())

        for topic_id in topic_ids:
            if topic_id not in self.questions:
                continue

            question = self.questions[topic_id]
            df = question.get_answers()

            if df.empty:
                continue

            df = df[[self.interview_col, self.line_col, self.speaker_col, self.text_col]].copy()
            df.rename(columns={
                self.interview_col: "interview_id",
                self.line_col: "line_id",
                self.speaker_col: "speaker_id",
                self.text_col: "text"
            }, inplace=True)

            if transcript_labels is not None:
                df.insert(1, "transcript_label", df["interview_id"].map(transcript_labels))

            label = slugify(question.label)
            filename = f"{topic_id}_{label}.csv"
            filepath = os.path.join(folder, filename)
            df.to_csv(filepath, index=False)

    def refit_outliers(self):
        mask = self.interviewer_lines['topic'] == -1
        texts = self.interviewer_lines.loc[mask, self.text_col].values
        new_topics, new_probs = self.model.transform(texts)

        new_topics = pd.Series(new_topics, index=self.interviewer_lines.index[mask])
        new_probs = pd.Series(new_probs, index=self.interviewer_lines.index[mask])

        for topic_id, question in self.questions.items():
            if question.has_children:
                sub_mask = new_topics == topic_id
                if not sub_mask.any():
                    continue
                sub_texts = self.interviewer_lines.loc[new_topics[sub_mask].index, self.text_col].to_list()
                sub_topics, sub_probs = question._split_state['model'].transform(sub_texts)

                offset = question._split_state['offset']
                sub_topics = pd.Series(sub_topics, index=new_topics[sub_mask].index) + offset
                sub_topics = sub_topics.replace(offset - 1, question.topic_id)

                sub_probs = pd.Series(sub_probs, index=new_probs[sub_mask].index)
                combined_probs = 2 * new_probs[sub_mask] * sub_probs / (new_probs[sub_mask] + sub_probs)

                new_topics[sub_mask] = sub_topics
                new_probs[sub_mask] = 2 * new_probs[sub_mask] * sub_probs / (new_probs[sub_mask] + sub_probs)

                new_topics[sub_mask] = sub_topics
                new_probs[sub_mask] = combined_probs

        self._refit_state = {
            'index': new_topics.index.tolist(),
            'topic': new_topics.to_list(),
            'prob': new_probs.to_list(),
            'text': texts
        }

        total = sum(mask)
        reassigned = sum(new_topics != -1)
        print(f"Reclassified {reassigned} out of {total} document(s). Use `examine_outliers()` to examine and `commit_outlier_refit()` to commit.")

    def examine_outliers(self, lower_prob=0, upper_prob=1, topics=None, return_tuples=False):
        if not hasattr(self, '_refit_state'):
            self.refit_outliers()

        df = pd.DataFrame(self._refit_state)
        df.set_index('index', inplace=True)

        name_dict = dict(zip(self.topic_info['Topic'], self.topic_info['Name']))
        topic_names = list(map(name_dict.get, self._refit_state['topic']))
        df.insert(loc=1, column='topic_name', value=topic_names)

        df = df[(df['prob'] >= lower_prob) & (df['prob'] <= upper_prob)]
        if topics is not None:
            df = df[df['topic'].isin(topics)]

        df.sort_values(by=['topic', 'prob'], ascending=[True, False], inplace=True)

        if return_tuples:
            return list(zip(df.index, df['topic_name'], df['prob'], df['text']))
        else:
            return df

    def commit_outlier_refit(self, lower_prob=0, upper_prob=1, topics=None):
        if not hasattr(self, '_refit_state'):
            raise RuntimeError("No reclassification found. Run `refit_outliers()` first.")

        df = pd.DataFrame(self._refit_state)
        df.set_index('index', inplace=True)

        df = df[(df['prob'] >= lower_prob) & (df['prob'] <= upper_prob)]
        if topics is not None:
            df = df[df['topic'].isin(topics)]

        if df.empty:
            print("No outliers matched the provided criteria. Nothing committed.")
            return

        self.interviewer_lines.loc[df.index, 'topic'] = df['topic'].values
        self.interviewer_lines.loc[df.index, 'topic_prob'] = df['prob'].values

        print(f"Committed {len(df)} reclassified outlier(s).")

    def export_outlier_refit(self, path, lower_prob=0, upper_prob=1, topics=None):
        if not hasattr(self, '_refit_state'):
            raise RuntimeError("No reclassification found. Run `refit_outliers()` first.")

        df = pd.DataFrame(self._refit_state)
        df.index = self._refit_state['index']

        name_dict = dict(zip(self.topic_info['Topic'], self.topic_info['Name']))
        topic_names = list(map(name_dict.get, self._refit_state['topic']))
        df.insert(loc=1, column='refit_topic_name', value=topic_names)

        df = df[(df['prob'] >= lower_prob) & (df['prob'] <= upper_prob)]
        if topics is not None:
            df = df[df['topic'].isin(topics)]

        df.sort_values(by=['topic', 'prob'], ascending=[True, False], inplace=True)

        df = df.rename(columns={
            'topic': 'refit_topic',
            'prob': 'refit_prob'
        })

        df['topic'] = -1
        df['manual_topic'] = ""
        df = df[['text', 'topic', 'refit_topic', 'refit_topic_name', 'refit_prob', 'manual_topic']]
        df.index.name = 'index'

        df.to_csv(path)
        print(f"Exported {len(df)} reclassified outlier(s) to {path}")

    def export_assignments(self, indexes, path):
        df = self.interviewer_lines.loc[indexes, [self.text_col, 'topic']].copy()
        df.index.name = 'index'

        name_dict = dict(zip(self.topic_info['Topic'], self.topic_info['Name']))
        df['topic_name'] = df['topic'].map(name_dict)
        df['manual_topic'] = ""

        # Reorder columns
        df = df[['text', 'topic', 'topic_name', 'manual_topic']]

        df.to_csv(path)
        print(f"Exported {len(df)} assignment row(s) to {path}")

    def manual_reassign(self, reassignments: dict, default_prob=1.0):
        """
        Manually reassign interviewer lines to existing topics.

        Parameters:
        - reassignments (dict): {row_index: topic_id} for rows in interviewer_lines
        - default_prob (float): Probability value to assign for manual reassignments
        """
        known_topics = set(self.questions.keys())
        valid = {idx: tid for idx, tid in reassignments.items() if tid in known_topics}
        invalid = {idx: tid for idx, tid in reassignments.items() if tid not in known_topics}

        if invalid:
            print(f"Ignored {len(invalid)} invalid reassignments to unknown topic IDs: {set(invalid.values())}")

        if not valid:
            print("No valid reassignments found.")
            return

        indices = list(valid.keys())
        topic_ids = list(valid.values())
        self.interviewer_lines.loc[indices, 'topic'] = topic_ids
        self.interviewer_lines.loc[indices, 'topic_prob'] = default_prob

        counts = self.interviewer_lines['topic'].value_counts()
        self.topic_info['Count'] = self.topic_info['Topic'].map(counts).fillna(0).astype('Int64')

        print(f"Manually reassigned {len(valid)} line(s).")

    def import_assignments(self, path, default_prob=1.0):
        df = pd.read_csv(path, index_col='index')

        if 'manual_topic' not in df.columns:
            raise ValueError("CSV must include a 'manual_topic' column.")

        df = df.dropna(subset=['manual_topic'])

        df['manual_topic'] = df['manual_topic'].astype(int)

        known = set(self.questions.keys())
        valid = df[df['manual_topic'].isin(known)]
        skipped = df[~df['manual_topic'].isin(known)]

        self.interviewer_lines.loc[valid.index, 'topic'] = valid['manual_topic'].values
        self.interviewer_lines.loc[valid.index, 'topic_prob'] = default_prob

        counts = self.interviewer_lines['topic'].value_counts()
        self.topic_info['Count'] = self.topic_info['Topic'].map(counts).fillna(0).astype('Int64')

        print(f"Imported {len(valid)} manual reassignment(s).")
        if not skipped.empty:
            bad_ids = sorted(skipped['manual_topic'].unique())
            print(f"Ignored {len(skipped)} invalid reassignment(s) to unknown topic(s): {bad_ids}")

    def export_interviews(self, folder,
                          transcript_labels=None,
                          speaker_labels=None,
                          speaker_prefix='Speaker_',
                          exclude_topics=None):
        os.makedirs(folder, exist_ok=True)
        exclude_topics = set(exclude_topics or [])

        df = self.label_full_data().copy()
        df.sort_values(by=[self.interview_col, self.line_col], inplace=True)

        if speaker_labels is not None:
            df[self.speaker_col] = df[self.speaker_col].map(speaker_labels).fillna(df[self.speaker_col])
        else:
            df[self.speaker_col] = df[self.speaker_col].map(lambda s: f"{speaker_prefix}{str(s)}")

        ignore_mask = [False] * len(df)
        if exclude_topics:
            for topic in exclude_topics:
                exclude_df = self.get_answers(topic, include_question=True, merge_transcript_lines=False)
                ignore_mask = df.index.isin(exclude_df.index) | ignore_mask
            df = df[~ignore_mask]

        for interview_id, group in df.groupby(self.interview_col):
            lines = []
            for _, row in group.iterrows():
                speaker = row[self.speaker_col]
                line = row[self.text_col].strip()
                if line:
                    lines.append(f"{speaker}: {line}")

            if transcript_labels is not None:
                name = transcript_labels[interview_id] if interview_id in transcript_labels else None
            else:
                name = ""
            filename = f"{interview_id}_{name}.txt"
            filepath = os.path.join(folder, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    def export_answer_visuals(self, folder, type='bar', use_tfidf=False, use_graph_topics=True, topics=None):
        os.makedirs(folder, exist_ok=True)
        if use_graph_topics:
            topic_ids = list(self.graph.nodes)
        else:
            topic_ids = topics if topics is not None else list(self.questions.keys())

        for topic_id in topic_ids:
            if topic_id not in self.questions:
                continue

            question = self.questions[topic_id]
            label = slugify(question.label)
            filename = f"{topic_id}_{label}.png"
            filepath = os.path.join(folder, filename)

            if type == 'bar':
                question.visualize_answers_bar(use_tfidf=use_tfidf, save_path=filepath)
            elif type == 'cloud':
                question.visualize_answers_cloud(use_tfidf=use_tfidf, save_path=filepath)
            else:
                raise ValueError(f"Unrecognized visual type: {type}.")

    def export_answer_lda_topics(self, folder,
                                 top_terms_n=10, top_docs_n=5, method="prob", transcript_labels=None,
                                 use_graph_topics=True, topics=None):
        os.makedirs(folder, exist_ok=True)
        if use_graph_topics:
            topic_ids = list(self.graph.nodes)
        else:
            topic_ids = topics if topics is not None else list(self.questions.keys())

        for topic_id in topic_ids:
            if topic_id not in self.questions:
                continue

            question = self.questions[topic_id]
            label = slugify(question.label)
            filename = f"{topic_id}_{label}.csv"
            filepath = os.path.join(folder, filename)

            df = question.get_answers_tm_lda_topics(top_terms_n=top_terms_n,
                                                    top_docs_n=top_docs_n,
                                                    method=method,
                                                    transcript_labels=transcript_labels)
            df.to_csv(filepath)


class ProtocolQuestion:
    def __init__(self, topic_id, lipe, parent=None):
        self.topic_id = topic_id
        self.lipe = lipe  # back-reference to parent
        if parent is not None:
            self.label = '_'.join([str(topic_id)] + self.representative_words[:4])
        self.ignore = False
        self._parent = parent
        self._family = {topic_id}

    @property
    def label(self):
        topic_info = self.lipe.topic_info
        return topic_info.loc[topic_info['Topic'] == self.topic_id, 'Name'].iloc[0]

    @label.setter
    def label(self, value):
        topic_info = self.lipe.topic_info
        topic_info.loc[topic_info['Topic'] == self.topic_id, 'Name'] = value

    @property
    def representative_words(self):
        topic_info = self.lipe.topic_info
        return topic_info.loc[topic_info['Topic'] == self.topic_id, 'Representation'].iloc[0]

    @property
    def count(self):
        topic_info = self.lipe.topic_info
        return topic_info.loc[topic_info['Topic'] == self.topic_id, 'Count'].iloc[0]

    def get_examples(self, n=5, show_processed=False, hide_probs=False):
        examples = self.lipe.detailed_examples(topics=[self.topic_id],
                                               n=n,
                                               show_processed=show_processed)['examples'].iloc[0]
        if hide_probs:
            examples = [text for _, text in examples]

        return examples

    def get_questions(self):
        mask = self.lipe.interviewer_lines['topic'] == self.topic_id
        texts = self.lipe.interviewer_lines.loc[mask, self.lipe.text_col].values
        return texts

    def get_answers(self, include_question=True, merge_transcript_lines=True):
        return self.lipe.get_answers(self.topic_id, include_question=include_question,
                                     merge_transcript_lines=merge_transcript_lines)

    @property
    def has_split(self):
        return hasattr(self, '_split_state')

    @property
    def children(self):
        return self._family - {self.topic_id}

    @property
    def has_children(self):
        return len(self.children) > 0

    @property
    def merges(self):
        return {k for k, v in self.lipe.question_merges.items() if v == self.topic_id}

    @property
    def descendants(self):
        to_check = self.children.copy()
        checked = {self.topic_id}

        while to_check:
            check = to_check.pop()
            if check not in checked:
                if self.lipe.questions[check].has_children:
                    to_check.update(self.lipe.questions[check].children)
                checked.add(check)
        return checked

    @property
    def merged_descendants(self):
        to_check = {self.topic_id}
        checked = set()

        while to_check:
            check = to_check.pop()
            if check not in checked:
                checked.add(check)
                to_check.update(self.lipe.questions[check].descendants)
                to_check.update(self.lipe.questions[check].merges)
        return checked

    @property
    def split_preview(self):
        if not self.has_split:
            return None
        return self._split_state['model'].get_topic_info()

    @property
    def root_topic(self):
        root = self.topic_id
        while root in self.lipe.question_merges:
            root = self.lipe.question_merges[root]
        return root

    def split(self, min_topic_size, custom_model=None, embedding_model_name=None, tm_preprocessor=None, random_state=42):
        # Extract relevant interviewer lines
        mask = self.lipe.interviewer_lines['topic'] == self.topic_id
        texts = self.lipe.interviewer_lines.loc[mask, self.lipe.text_col]

        if len(texts) < 1:
            raise ValueError(f"No documents found for topic {self.topic_id}.")

        # Use user-provided or new model
        if custom_model:
            model = custom_model
        else:
            embedding_model_name = embedding_model_name or self.lipe.embedding_model_name
            embed_model = SentenceTransformer(embedding_model_name)
            umap_model = UMAP(n_neighbors=min_topic_size, n_components=5, min_dist=0.0,
                              metric='cosine', random_state=random_state)
            hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size,
                                    min_samples=max(1, min_topic_size // 2),
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True)
            model = BERTopic(embedding_model=embed_model,
                             umap_model=umap_model,
                             hdbscan_model=hdbscan_model,
                             min_topic_size=min_topic_size)

        if tm_preprocessor:
            texts = texts.apply(tm_preprocessor)
        new_topics, new_probs = model.fit_transform(texts.to_list())

        self._split_state = {
            'mask': mask,
            'model': model,
            'new_topics': new_topics,
            'new_probs': new_probs,
            'docs': texts
        }
        return self._split_state['model'].get_topic_info()

    def get_split_examples(self, topic, n=10):
        docs = [d for t, d in zip(self._split_state['new_topics'], self._split_state['docs']) if t == topic]
        if len(docs) > n:
            docs = random.sample(docs, n)
        return docs

    def visualize_split_documents(self):
        docs = self.lipe.interviewer_lines[self._split_state['mask']][self.lipe.text_col].to_list()
        return self._split_state['model'].visualize_documents(docs)

# WIP
#    def reduce_split_outliers(self, strategy='distributions', final=False):
#        new_topics = topic_model.reduce_outliers(self._split_state['docs'], self._split_state['new_topics'], strategy=strategy)

    def commit_split(self):
        """
        Register the split topics as part of LIPE while keeping the old topics
        """
        if not hasattr(self, '_split_state'):
            raise RuntimeError("No split has been run. Call `.split()` first.")
        elif self.has_children:
            raise RuntimeError(f"Question is already split into questions: {', '.join([str(i) for i in self.children])}")

        # Update LIPE topic labels and probabilities (using harmonic mean)
        next_index = self.lipe.next_question_index()
        self._split_state['offset'] = next_index
        adjusted_topics = pd.Series(self._split_state['new_topics']) + next_index
        adjusted_topics = adjusted_topics.replace(next_index - 1, self.topic_id)

        old_probs = self.lipe.interviewer_lines.loc[self._split_state['mask'], 'topic_prob'].values
        self._split_state['old_probs'] = old_probs
        new_probs = np.asarray(self._split_state['new_probs'])
        adjusted_probs = 2 * old_probs * new_probs / (old_probs + new_probs)

        self.lipe.interviewer_lines.loc[self._split_state['mask'], 'topic'] = adjusted_topics.values
        self.lipe.interviewer_lines.loc[self._split_state['mask'], 'topic_prob'] = adjusted_probs

        # Update question children
        self._family = set(adjusted_topics)

        # Update LIPE.topic_info
        new_topic_info = self._split_state['model'].get_topic_info().copy()
        if new_topic_info.iloc[0, 0] == -1:
            new_topic_info.drop(index=0, inplace=True)
        new_topic_info['Topic'] += next_index
        new_topic_info['Parent_Topic'] = self.topic_id

        self.lipe.topic_info = pd.concat([self.lipe.topic_info, new_topic_info], ignore_index=True)

        # Update LIPE.questions
        for topic_id in new_topic_info['Topic']:
            self.lipe.questions[topic_id] = ProtocolQuestion(topic_id, self.lipe, self.topic_id)

    def unsplit(self):
        if not self.has_children:
            raise RuntimeError("Question currently has no children to unsplit")

        self.lipe.interviewer_lines.loc[self._split_state['mask'], 'topic'] = self.topic_id
        self.lipe.interviewer_lines.loc[self._split_state['mask'], 'topic_prob'] = self._split_state['old_probs']
        del self._split_state['old_probs']
        del self._split_state['offset']
        to_remove = self.children
        self._family = {self.topic_id}
        self.lipe.topic_info = self.lipe.topic_info[~self.lipe.topic_info['Topic'].isin(to_remove)]
        self.lipe.question_merges = {k: v for k, v in self.lipe.question_merges.items() if not (k in to_remove or v in to_remove)}
        for i in to_remove:
            del self.lipe.questions[i]

    def merge_with(self, topic_id):
        loop = forms_loop(self.lipe.question_merges, self.topic_id, topic_id)
        if loop:
            raise RuntimeError(f"Merge would form a loop: {loop}")
        else:
            self.lipe.question_merges.update({self.topic_id: topic_id})

    def unmerge(self):
        if self.topic_id in self.lipe.question_merges:
            parent = self.lipe.question_merges.pop(self.topic_id)
            print(f"Question {self.topic_id} unmerged from {parent}")
        else:
            raise RuntimeError(f"Question {self.topic_id} is not merged")

    def get_answer_dtm(self,
                       lemmatize=True,
                       lowercase=True,
                       remove_numbers=True,
                       remove_punctuation=True,
                       custom_stopwords=None,
                       keep_words=None):
        df = self.get_answers(include_question=False, merge_transcript_lines=True)

        if not hasattr(self.lipe, "spacy_model"):
            self.lipe.load_spacy()
        nlp = self.lipe.spacy_model

        custom_stopwords = set(custom_stopwords or [])
        keep_words = set(keep_words or [])

        def token_form(token):
            text = token.lemma_ if lemmatize else token.text
            return text.lower() if lowercase else text

        def spacy_tokenizer(text):
            doc = nlp(text)
            return [
                t for token in doc for t in [token_form(token)]
                if (
                    not (remove_numbers and token.is_digit)
                    and not (remove_punctuation and token.is_punct)
                    and not ((t in custom_stopwords) or (token.is_stop and t not in keep_words))
                )
            ]

        vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, lowercase=False)
        dtm = vectorizer.fit_transform(df[self.lipe.text_col])

        self._answer_dtm = dtm
        self._answer_dtm_index = df[self.lipe.interview_col].tolist()
        self._answer_vocab = vectorizer.get_feature_names_out()
        self._answer_vectorizer = vectorizer
        return dtm

    def get_answer_tf_idf(self):
        if not hasattr(self, "_answer_dtm"):
            self.get_answer_dtm()
        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(self._answer_dtm)
        return tfidf_matrix

    def get_top_words(self, top_n=20, sort_by='count', ascending=False):
        if not hasattr(self, "_answer_dtm"):
            self.get_answer_dtm()

        tfidf = self.get_answer_tf_idf()
        n_docs = self._answer_dtm.shape[0]

        # Total term frequency (across all docs)
        counts = self._answer_dtm.sum(axis=0).A1

        # Document frequency (non-zero rows per column)
        doc_freq = np.array((self._answer_dtm > 0).sum(axis=0)).ravel()
        doc_prop = doc_freq / n_docs

        # TF-IDF summed over all docs
        tfidf_sum = tfidf.sum(axis=0).A1

        df = pd.DataFrame({
            'word': self._answer_vocab,
            'count': counts,
            'doc_freq': doc_freq,
            'doc_prop': doc_prop,
            'tfidf': tfidf_sum
        })

        df = df.sort_values(by=sort_by, ascending=ascending).head(top_n)
        return df.reset_index(drop=True)

    def visualize_answers_cloud(self,
                                use_tfidf=False,
                                max_words=100,
                                width=800,
                                height=400,
                                background_color='white',
                                colormap=None,
                                prefer_horizontal=0.9,
                                scale=1,
                                relative_scaling='auto',
                                normalize_plurals=True,
                                contour_color='black',
                                contour_width=0,
                                save_path=None):
        if use_tfidf:
            matrix = self.get_answer_tf_idf()
        else:
            if not hasattr(self, "_answer_dtm"):
                self.get_answer_dtm()
            matrix = self._answer_dtm

        weights = matrix.sum(axis=0).A1
        freqs = dict(zip(self._answer_vocab, weights))

        wc = WordCloud(width=width,
                       height=height,
                       background_color=background_color,
                       max_words=max_words,
                       colormap=colormap,
                       prefer_horizontal=prefer_horizontal,
                       scale=scale,
                       relative_scaling=relative_scaling,
                       normalize_plurals=normalize_plurals,
                       contour_color=contour_color,
                       contour_width=contour_width
                       ).generate_from_frequencies(freqs)

        plt.figure(figsize=(width / 100, height / 100))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        label = self.label if hasattr(self, "label") else self.topic_id
        plt.title(f"Word Cloud for Question {label}")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_answers_bar(self, top_n=20, use_tfidf=False,
                              figsize=(10, 6),
                              color='skyblue',
                              save_path=None):
        if use_tfidf:
            matrix = self.get_answer_tf_idf()
        else:
            if not hasattr(self, "_answer_dtm"):
                self.get_answer_dtm()
            matrix = self._answer_dtm

        weights = matrix.sum(axis=0).A1
        freqs = pd.Series(weights, index=self._answer_vocab)
        top_words = freqs.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=figsize)
        top_words.sort_values().plot(kind="barh", color=color)
        label = self.label if hasattr(self, "label") else self.topic_id
        plt.title(f"Top {top_n} Words for Question {label}")
        plt.xlabel("TF-IDF Score" if use_tfidf else "Count")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def fit_answers_tm_bertopic(self, embedding_model=None, processor=None,
                                min_topic_size=5, spacy_language_code="en"):
        answers_df = self.get_answers(include_question=False, merge_transcript_lines=True)

        nlp = spacy.blank(spacy_language_code)
        nlp.add_pipe("sentencizer")

        all_sentences = []
        for _, row in answers_df.iterrows():
            doc = nlp(row[self.lipe.text_col])
            all_sentences.extend([
                (row[self.lipe.interview_col], i, sent.text.strip())
                for i, sent in enumerate(doc.sents)
            ])
        sent_df = pd.DataFrame(all_sentences, columns=["transcript_id", "sentence_id", "sentence"])

        if embedding_model is None:
            embedding_model = SentenceTransformer("intfloat/e5-base-v2")
            if processor is None:
                processor = e5_preprocessor

        texts = sent_df["sentence"]
        if processor is not None:
            texts = texts.apply(processor)

        tm = BERTopic(embedding_model=embedding_model, min_topic_size=min_topic_size)
        topics, probs = tm.fit_transform(texts.tolist())
        sent_df["topic"] = topics

        self.answers_sentences = sent_df
        self.answers_bertopic = tm

        return tm.get_topic_info()

    def fit_answers_tm_lda(self, n_topics=10):
        if not hasattr(self, "_answer_dtm") or self._answer_dtm is None:
            self.get_answer_dtm()

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        doc_topic_dist = lda.fit_transform(self._answer_dtm)
        topic_word_dist = lda.components_

        self.answer_tm_lda_doc_topic_dist = doc_topic_dist
        self.answer_tm_lda_topic_word_dist = topic_word_dist

        return self.get_answers_tm_lda_topics()

    def get_answers_tm_lda_docs(self, top_k=3):
        if not hasattr(self, "answer_tm_lda_doc_topic_dist") or self.answer_tm_lda_doc_topic_dist is None:
            self.fit_answers_tm_lda()

        dist = self.answer_tm_lda_doc_topic_dist
        doc_ids = self._answer_dtm_index

        top_indices = np.argsort(-dist, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(dist, top_indices, axis=1)

        data = []
        for i, doc_id in enumerate(doc_ids):
            row = {"doc_id": doc_id}
            for j in range(top_k):
                row[f"topic_{j+1}"] = top_indices[i, j]
                row[f"score_{j+1}"] = top_scores[i, j]
            data.append(row)

        return pd.DataFrame(data)

    def get_answers_tm_lda_topics(self, top_terms_n=10, top_docs_n=5, method="prob", transcript_labels=None):
        if not hasattr(self, "answer_tm_lda_doc_topic_dist") or self.answer_tm_lda_doc_topic_dist is None:
            self.fit_answers_tm_lda()

        topic_word = self.answer_tm_lda_topic_word_dist
        doc_topic = self.answer_tm_lda_doc_topic_dist
        vocab = self._answer_vocab
        doc_ids = np.array(self._answer_dtm_index)

        n_topics = topic_word.shape[0]
        word_scores = topic_word if method == "prob" else None

        # Top topic per doc
        top_topic_ids = doc_topic.argmax(axis=1)
        topic_counts = pd.Series(top_topic_ids).value_counts().sort_index()
        avg_prevalence = doc_topic.mean(axis=0)

        # Compute exclusivity and FREX if needed
        if method == "frex":
            word_totals = topic_word.sum(axis=0)
            exclusivity = topic_word / (word_totals + 1e-10)
            word_scores = 2 * topic_word * exclusivity / (topic_word + exclusivity + 1e-10)

        # Build top word lists with scores
        top_words = {}
        for i in range(n_topics):
            scores = word_scores[i]
            top_idx = scores.argsort()[::-1][:top_terms_n]
            top_words[i] = [(vocab[j], float(scores[j])) for j in top_idx]

        # Build top doc lists with labels and probs
        top_docs = {}
        for i in range(n_topics):
            probs = doc_topic[:, i]
            top_idx = probs.argsort()[::-1][:top_docs_n]
            top_docs[i] = [
                (transcript_labels.get(doc_ids[j], doc_ids[j]) if transcript_labels is not None else doc_ids[j], float(probs[j]))
                for j in top_idx
            ]

        # Combine results
        df = pd.DataFrame({
            "topic": range(n_topics),
            "num_docs": [topic_counts.get(i, 0) for i in range(n_topics)],
            "avg_prevalence": avg_prevalence,
            "top_words": [top_words[i] for i in range(n_topics)],
            "top_docs": [top_docs[i] for i in range(n_topics)]
        })

        return df.sort_values('avg_prevalence', ascending=False)

    def visualize_answers_tm_bertopic(self):
        if not hasattr(self, "answer_bertopic"):
            raise RuntimeError("No BERTopic model fitted yet. Run `fit_answers_tm_bertopic()` first.")

        return self.answers_bertopic.visualize_documents(self.answers_sentences.sentence)


def forms_loop(links, origin, target):
    path = f"{origin} -> {target}"
    if target == origin:
        return path
    while target in links:
        target = links[target]
        path += " -> " + str(target)
        if target == origin:
            return path
    return False
