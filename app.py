import streamlit as st
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import textwrap

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .main .block-container {padding: 1rem; max-width: 100vw;}
        header, footer {visibility: hidden;}
        .element-container:has(.stComponent iframe) {padding: 0 !important; border: none !important; box-shadow: none !important; background: transparent !important;}
        iframe {background: #0e1117 !important;}
        .graph-area-moveup {margin-top: -32px !important;}
        textarea, .stTextArea textarea {min-height: 600px !important; max-height: 600px !important;}
        .centered-header {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

nltk.download('punkt')
nltk.download('punkt_tab')

if "paragraphs" not in st.session_state:
    st.session_state.paragraphs = [
        "Quantum entanglement, a phenomenon wherein particles become interconnected such that the state of one instantaneously influences the state of another regardless of distance, challenges classical intuitions about locality and causality. The Einstein-Podolsky-Rosen paradox questioned whether quantum mechanics could provide a complete description of physical reality, suggesting the existence of hidden variables. Bell's theorem, however, demonstrated that no local hidden variable theory can reproduce all predictions of quantum mechanics, a result confirmed by numerous experiments. Despite these findings, the interpretation of entanglement remains a subject of philosophical debate, particularly concerning the nature of reality and information transfer."
    ]
if "graphs" not in st.session_state:
    st.session_state.graphs = [None for _ in st.session_state.paragraphs]
if "top_indices_list" not in st.session_state:
    st.session_state.top_indices_list = [None for _ in st.session_state.paragraphs]
if "sentence_summaries_list" not in st.session_state:
    st.session_state.sentence_summaries_list = [None for _ in st.session_state.paragraphs]
if "graph_generated" not in st.session_state:
    st.session_state.graph_generated = [False for _ in st.session_state.paragraphs]
if "show_combined_graph" not in st.session_state:
    st.session_state.show_combined_graph = False
if "combined_nodes_cache" not in st.session_state:
    st.session_state.combined_nodes_cache = None

def wrap_label(text, width=35):
    return "\n".join(textwrap.wrap(text, width=width))

def get_best_paths_all_sources(precedence_graph, n, max_sim=1.01):
    covered_targets = set()
    rows = []
    for source in range(n):
        for target in range(n):
            if target == source or target in covered_targets:
                continue
            try:
                path = nx.dijkstra_path(precedence_graph, source, target, weight=lambda u, v, d: max_sim - d['weight'])
                weights = [precedence_graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])]
                avg_sim = np.mean(weights) if weights else 0.0
                rows.append({
                    "From": source,
                    "To": target,
                    "Reading Direction": " → ".join(str(num) for num in path),
                    "Average Similarity": round(avg_sim, 3)
                })
                covered_targets.add(target)
            except nx.NetworkXNoPath:
                continue
            except Exception:
                continue
        if len(covered_targets) == n - 1:
            break
    if rows and "Average Similarity" in rows[0]:
        df = pd.DataFrame(rows)
        df = df[df["Average Similarity"].apply(lambda x: isinstance(x, float) or isinstance(x, int))]
        return df
    else:
        return pd.DataFrame(rows)

st.title("Tapestra: The Concept Graph")
for idx in range(len(st.session_state.paragraphs)):
    col_left, col_right, col_physics = st.columns([1, 3, 1], gap="large")

    # --- PHYSICS CONTROLS FIRST, SO VARIABLES ARE AVAILABLE ---
    with col_physics:
        st.header("Physics Controls")
        spring_length = st.slider("Spring Length (gap between nodes)", min_value=100, max_value=1200, value=500, step=50, key=f"spring_length_{idx}")
        spring_constant = st.slider("Spring Constant (lower = more flexible)", min_value=0.001, max_value=0.05, value=0.005, step=0.001, format="%.3f", key=f"spring_constant_{idx}")
        grav_constant = st.slider("Gravitational Constant (less negative = less clustering)", min_value=-5000, max_value=-100, value=-1000, step=100, key=f"grav_constant_{idx}")
        central_gravity = st.slider("Central Gravity", min_value=0.0, max_value=1.0, value=0.3, step=0.05, key=f"central_gravity_{idx}")

    with col_left:
        st.header("Type into the space below to graph_")
        st.markdown(f"**Paragraph {idx+1}**")
        gen_btn = st.button("Generate Graph", key=f"gen_graph_{idx}")
        old_text = st.session_state.paragraphs[idx]
        new_text = st.text_area(
            "", value=old_text, height=600, key=f"para_{idx}"
        )
        if new_text != old_text:
            st.session_state.paragraphs[idx] = new_text
            st.session_state.graphs[idx] = None
            st.session_state.top_indices_list[idx] = None
            st.session_state.sentence_summaries_list[idx] = None
            st.session_state.graph_generated[idx] = False
            st.session_state.show_combined_graph = False
            st.session_state.combined_nodes_cache = None
        if st.button("Add Paragraph", key=f"add_paragraph_btn_{idx}"):
            insert_idx = idx + 1
            st.session_state.paragraphs.insert(insert_idx, "")
            st.session_state.graphs.insert(insert_idx, None)
            st.session_state.top_indices_list.insert(insert_idx, None)
            st.session_state.sentence_summaries_list.insert(insert_idx, None)
            st.session_state.graph_generated.insert(insert_idx, False)
            st.session_state.show_combined_graph = False
            st.session_state.combined_nodes_cache = None
            st.rerun()

    with col_right:
        graph_path = f"graph_{idx}.html"
        error_msg = None
        if gen_btn:
            try:
                paragraph = st.session_state.paragraphs[idx]
                sentences = nltk.sent_tokenize(paragraph)
                n = len(sentences)
                sentence_summaries = sentences
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(sentences)
                tfidf_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
                position_scores = np.zeros(n)
                position_scores[0] += 0.2
                position_scores[-1] += 0.2
                similarity_matrix = cosine_similarity(tfidf_matrix)
                np.fill_diagonal(similarity_matrix, 0)
                G = nx.from_numpy_array(similarity_matrix)
                pagerank_scores = np.array([score for _, score in nx.pagerank(G).items()])
                combined_scores = tfidf_scores + position_scores + pagerank_scores
                top_indices = combined_scores.argsort()[-2:][::-1]
                precedence_graph = nx.DiGraph()
                precedence_graph.add_nodes_from(range(n))
                threshold = 0.15
                for i in range(n):
                    for j in range(i + 1, n):
                        if similarity_matrix[i, j] > threshold:
                            precedence_graph.add_edge(i, j, weight=similarity_matrix[i, j])
                for i in range(n):
                    if i not in top_indices:
                        similarities_to_core = [(core_idx, similarity_matrix[i, core_idx]) for core_idx in top_indices if i < core_idx]
                        if similarities_to_core:
                            core_idx, max_sim = max(similarities_to_core, key=lambda x: x[1])
                            if max_sim > threshold and not precedence_graph.has_edge(i, core_idx):
                                precedence_graph.add_edge(i, core_idx, weight=max_sim)
                st.session_state.graphs[idx] = precedence_graph
                st.session_state.top_indices_list[idx] = top_indices
                st.session_state.sentence_summaries_list[idx] = sentence_summaries
                st.session_state.graph_generated[idx] = True
                st.session_state.show_combined_graph = False
                st.session_state.combined_nodes_cache = None
            except Exception as e:
                error_msg = f"Error generating graph: {str(e)}"
        if st.session_state.graph_generated[idx] and st.session_state.graphs[idx] is not None:
            precedence_graph = st.session_state.graphs[idx]
            top_indices = st.session_state.top_indices_list[idx]
            sentence_summaries = st.session_state.sentence_summaries_list[idx]
            net = Network(height="700px", width="100%", directed=True, notebook=False, bgcolor="#0e1117")
            net.barnes_hut()
            for i, summary in enumerate(sentence_summaries):
                node_label = f"[{i}]\n{wrap_label(summary, width=35)}"
                if i == 0:
                    net.add_node(
                        i,
                        label=node_label,
                        title=summary,
                        color={"background": "#FFFF99", "border": "#FFD700", "highlight": {"background": "#FFD700", "border": "#FFD700"}},
                        shape="circle",
                        font={"size": 20, "face": "arial", "multi": True, "color": "#111111", "bold": True},
                        borderWidth=5
                    )
                elif i in top_indices:
                    net.add_node(
                        i,
                        label=node_label,
                        title=summary,
                        color={"background": "#FFA500", "border": "#FF8C00", "highlight": {"background": "#FFD580", "border": "#FF8C00"}},
                        shape="box",
                        font={"size": 20, "face": "arial", "multi": True, "color": "#111111", "bold": True},
                        widthConstraint={"maximum": 400, "minimum": 200}
                    )
                else:
                    net.add_node(
                        i,
                        label=node_label,
                        title=summary,
                        color={"background": "#87CEFA", "border": "#4682B4", "highlight": {"background": "#B0E0E6", "border": "#4682B4"}},
                        shape="box",
                        font={"size": 20, "face": "arial", "multi": True, "color": "#111111", "bold": True},
                        widthConstraint={"maximum": 400, "minimum": 200}
                    )
            for u, v, w in precedence_graph.edges(data='weight'):
                net.add_edge(
                    u, v, value=w, arrowStrikethrough=False, arrows="to",
                    color="#FFFFFF", width=2,
                    smooth={"type": "curvedCW"},
                    title=f"Similarity: {w:.2f}"
                )
            net.set_options(f"""
            {{
              "edges": {{
                "arrows": {{
                  "to": {{
                    "enabled": true,
                    "type": "arrow",
                    "scaleFactor": 2.2,
                    "color": "#FFFFFF"
                  }}
                }},
                "color": "#FFFFFF",
                "smooth": {{
                  "type": "curvedCW",
                  "roundness": 0.3
                }}
              }},
              "nodes": {{
                "borderWidth": 2,
                "shadow": true,
                "font": {{
                  "size": 20,
                  "face": "arial",
                  "multi": true,
                  "color": "#111111",
                  "bold": true
                }},
                "widthConstraint": {{
                  "maximum": 400,
                  "minimum": 200
                }}
              }},
              "physics": {{
                "barnesHut": {{
                  "gravitationalConstant": {grav_constant},
                  "centralGravity": {central_gravity},
                  "springLength": {spring_length},
                  "springConstant": {spring_constant}
                }},
                "minVelocity": 0.75,
                "timestep": 0.2,
                "stabilization": {{
                  "enabled": true,
                  "iterations": 2000,
                  "fit": true
                }}
              }},
              "interaction": {{
                "zoomView": true,
                "dragView": true,
                "dragNodes": true,
                "multiselect": true,
                "navigationButtons": true,
                "keyboard": true
              }},
              "layout": {{
                "improvedLayout": true
              }},
              "autoResize": true,
              "height": "100%",
              "width": "100%",
              "background": "#0e1117"
            }}
            """)
            net.save_graph(graph_path)
            st.components.v1.html(open(graph_path, "r", encoding="utf-8").read(), height=700, scrolling=True)
            with open(graph_path, "rb") as f:
                st.download_button(
                    label=f"Download Graph for Paragraph {idx+1} (Open in Browser)",
                    data=f,
                    file_name=f"graph_{idx}.html",
                    mime="text/html"
                )
            st.markdown("#### Optimal Para Sequencing")
            try:
                df_paths = get_best_paths_all_sources(precedence_graph, len(sentence_summaries))
                if "Average Similarity" in df_paths.columns:
                    df_paths = df_paths[df_paths["Average Similarity"].apply(lambda x: isinstance(x, float) or isinstance(x, int))]
                    if not df_paths.empty:
                        st.table(df_paths)
                    else:
                        st.warning("None of the lines can be interlinked.")
                else:
                    st.warning("None of the lines can be interlinked.")
            except Exception as e:
                st.error(f"Error in path calculation: {str(e)}")
        if error_msg:
            st.error(error_msg)
    st.markdown("---")

if st.button("Article Covered: Combine Information", key="combine_info_btn"):
    st.session_state.show_combined_graph = True
    combined_nodes = []
    node_id = 0
    combined_texts = []
    for para_idx, (graph, top_indices, summaries, generated) in enumerate(
        zip(
            st.session_state.graphs,
            st.session_state.top_indices_list,
            st.session_state.sentence_summaries_list,
            st.session_state.graph_generated,
        )
    ):
        if not generated or not graph:
            continue
        for i, summary in enumerate(summaries):
            if i == 0 or (top_indices is not None and i in top_indices):
                combined_nodes.append((node_id, summary, i == 0))
                combined_texts.append(summary)
                node_id += 1
    st.session_state.combined_nodes_cache = (combined_nodes, combined_texts)

if st.session_state.show_combined_graph and st.session_state.combined_nodes_cache:
    st.header("Combined Concept Graph (All Paragraphs)")
    combined_nodes, combined_texts = st.session_state.combined_nodes_cache
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)
    n_combined = len(combined_nodes)
    combined_graph_nx = nx.DiGraph()
    for i in range(n_combined):
        combined_graph_nx.add_node(i)
    threshold = 0.15
    for i in range(n_combined):
        for j in range(n_combined):
            if i != j and similarity_matrix[i, j] > threshold:
                combined_graph_nx.add_edge(i, j, weight=similarity_matrix[i, j])
    combined_net = Network(height="1000px", width="100%", directed=True, notebook=False, bgcolor="#0e1117")
    combined_net.barnes_hut()
    for idx, (nid, summary, is_start) in enumerate(combined_nodes):
        node_label = f"[{nid}]\n{wrap_label(summary, width=35)}"
        if idx == 0:
            combined_net.add_node(
                nid,
                label=node_label,
                title=summary,
                color={"background": "#FFFF99", "border": "#FFD700", "highlight": {"background": "#FFD700", "border": "#FFD700"}},
                shape="circle",
                font={"size": 22, "face": "arial", "multi": True, "color": "#111111", "bold": True},
                borderWidth=5
            )
        else:
            combined_net.add_node(
                nid,
                label=node_label,
                title=summary,
                color={"background": "#FFA500", "border": "#FF8C00", "highlight": {"background": "#FFD580", "border": "#FF8C00"}},
                shape="box",
                font={"size": 22, "face": "arial", "multi": True, "color": "#111111", "bold": True},
                widthConstraint={"maximum": 400, "minimum": 200}
            )
    for u, v, w in combined_graph_nx.edges(data='weight'):
        combined_net.add_edge(
            u, v, value=w, arrowStrikethrough=False, arrows="to",
            color="#FFFFFF", width=2,
            smooth={"type": "curvedCW"},
            title=f"Similarity: {w:.2f}"
        )
    combined_net.set_options(f"""
    {{
      "edges": {{
        "arrows": {{
          "to": {{
            "enabled": true,
            "type": "arrow",
            "scaleFactor": 2.2,
            "color": "#FFFFFF"
          }}
        }},
        "color": "#FFFFFF",
        "smooth": {{
          "type": "curvedCW",
          "roundness": 0.3
        }}
      }},
      "nodes": {{
        "borderWidth": 2,
        "shadow": true,
        "font": {{
          "size": 22,
          "face": "arial",
          "multi": true,
          "color": "#111111",
          "bold": true
        }},
        "widthConstraint": {{
          "maximum": 400,
          "minimum": 200
        }}
      }},
      "physics": {{
        "barnesHut": {{
          "gravitationalConstant": {grav_constant},
          "centralGravity": {central_gravity},
          "springLength": {spring_length},
          "springConstant": {spring_constant}
        }},
        "minVelocity": 0.75,
        "timestep": 0.2,
        "stabilization": {{
          "enabled": true,
          "iterations": 2000,
          "fit": true
        }}
      }},
      "interaction": {{
        "zoomView": true,
        "dragView": true,
        "dragNodes": true,
        "multiselect": true,
        "navigationButtons": true,
        "keyboard": true
      }},
      "layout": {{
        "improvedLayout": true
      }},
      "autoResize": true,
      "height": "100%",
      "width": "100%",
      "background": "#0e1117"
    }}
    """)
    combined_path = "combined_graph.html"
    combined_net.save_graph(combined_path)
    st.components.v1.html(open(combined_path, "r", encoding="utf-8").read(), height=1000, scrolling=True)
    with open(combined_path, "rb") as f:
        st.download_button(
            label="Download Idea Map (Open in Browser)",
            data=f,
            file_name="combined_graph.html",
            mime="text/html"
        )
    st.markdown("#### Article Walkthrough")
    try:
        df_combined = get_best_paths_all_sources(combined_graph_nx, n_combined)
        if "Average Similarity" in df_combined.columns:
            df_combined = df_combined[df_combined["Average Similarity"].apply(lambda x: isinstance(x, float) or isinstance(x, int))]
            st.table(df_combined)
        else:
            st.warning("The ideas are not interdependent. All key sentences require separate reading.")
    except Exception as e:
        st.error(f"Error in combined graph path calculation: {str(e)}")
    st.info("All start and key nodes from all paragraphs are combined in topological order. First node is yellow, rest are orange.")
