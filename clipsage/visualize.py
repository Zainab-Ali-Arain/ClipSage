"""
Stage 6 - Visual representation.

  * concept map: key terms that appear together (in the same or neighbouring
    sentences) are linked, giving a co-occurrence network (NetworkX), drawn
    interactively with PyVis and statically with Matplotlib for the report
  * keyword cloud (WordCloud)
  * timeline of the best video segments (Plotly)
"""

from itertools import combinations
from typing import Dict, List, Tuple

import networkx as nx

from . import config
from .text import contains_term, normalize_query
from .utils import format_timestamp, youtube_link

QUERY_COLOR = "#c2410c"
TERM_COLOR = "#2563eb"


def build_concept_graph(query: str, key_terms: List[Tuple[str, float]], hits: List[dict],
                        n_terms: int = config.CONCEPT_MAP_TERMS,
                        max_edges_per_term: int = 3) -> nx.Graph:
    topic = normalize_query(query).lower()
    terms = [t for t, _ in key_terms[:n_terms]]
    weights = dict(key_terms)

    # "windows" = each sentence together with its neighbour, inside each segment
    windows: List[str] = []
    for h in hits:
        sents = [s["text"] for s in h.get("sentences", [])] or [h["text"]]
        for i in range(len(sents)):
            windows.append(" ".join(sents[i:i + 2]))

    present = {t: [contains_term(w, t) for w in windows] for t in terms}
    topic_in = [contains_term(w, topic) for w in windows]

    g = nx.Graph()
    g.add_node(topic, kind="query", weight=1.0)
    for t in terms:
        g.add_node(t, kind="term", weight=float(weights.get(t, 0.5)))

    # every term is linked to the topic; the weight is how often they co-occur
    for t in terms:
        co = sum(a and b for a, b in zip(present[t], topic_in))
        g.add_edge(topic, t, weight=max(1, co), cooccurrence=co)

    # term-term links, keeping only the strongest few per term to stay readable
    pair_counts: Dict[Tuple[str, str], int] = {}
    for a, b in combinations(terms, 2):
        c = sum(x and y for x, y in zip(present[a], present[b]))
        if c >= config.MIN_COOCCURRENCE:
            pair_counts[(a, b)] = c
    degree: Dict[str, int] = {t: 0 for t in terms}
    for (a, b), c in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        if degree[a] < max_edges_per_term and degree[b] < max_edges_per_term:
            g.add_edge(a, b, weight=c, cooccurrence=c)
            degree[a] += 1
            degree[b] += 1
    return g


def concept_map_html(g: nx.Graph, height: str = "520px") -> str:
    """Interactive PyVis concept map as a self-contained HTML string."""
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#1f2937",
                  cdn_resources="in_line", notebook=False)
    for node, data in g.nodes(data=True):
        is_query = data.get("kind") == "query"
        net.add_node(node, label=node, title=node,
                     color=QUERY_COLOR if is_query else TERM_COLOR,
                     size=34 if is_query else 12 + 18 * data.get("weight", 0.5),
                     font={"size": 22 if is_query else 16})
    for a, b, data in g.edges(data=True):
        net.add_edge(a, b, value=data.get("weight", 1),
                     title=f"appear together {data.get('cooccurrence', 0)} time(s)",
                     color="#9ca3af")
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -6000, "springLength": 140}},'
                    ' "interaction": {"hover": true}}')
    return net.generate_html()


def concept_map_figure(g: nx.Graph):
    """Static Matplotlib concept map (for the report / slides)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(g, seed=7, k=0.9)
    colors = [QUERY_COLOR if d.get("kind") == "query" else TERM_COLOR for _, d in g.nodes(data=True)]
    sizes = [2600 if d.get("kind") == "query" else 700 + 1500 * d.get("weight", 0.5) for _, d in g.nodes(data=True)]
    widths = [0.8 + 0.6 * d.get("weight", 1) for _, _, d in g.edges(data=True)]
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, edge_color="#9ca3af", alpha=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=10, font_color="#111827")
    ax.axis("off")
    fig.tight_layout()
    return fig


def keyword_cloud(key_terms: List[Tuple[str, float]], width: int = 900, height: int = 420):
    """WordCloud image (PIL) from term weights."""
    from wordcloud import WordCloud

    freqs = {t: max(s, 0.01) for t, s in key_terms}
    if not freqs:
        return None
    wc = WordCloud(width=width, height=height, background_color="white", colormap="viridis",
                   prefer_horizontal=0.9, random_state=7)
    return wc.generate_from_frequencies(freqs).to_image()


def timeline_figure(hits: List[dict]):
    """Plotly timeline: where in each video the best segments are."""
    import plotly.graph_objects as go

    fig = go.Figure()
    titles: List[str] = []
    for h in hits:
        label = (h["title"][:45] + "...") if len(h["title"]) > 48 else h["title"]
        label = f"{label} ({h['channel_title']})" if h.get("channel_title") else label
        if label not in titles:
            titles.append(label)
        fig.add_trace(go.Bar(
            x=[(h["end"] - h["start"]) / 60], base=[h["start"] / 60], y=[label], orientation="h",
            marker=dict(color=h["score"], colorscale="Blues", cmin=0, cmax=max(x["score"] for x in hits) or 1),
            hovertemplate=(f"<b>#{h['rank']}</b> {format_timestamp(h['start'])}-{format_timestamp(h['end'])}"
                           f"<br>similarity {h['score']:.2f}<br>{h['text'][:140]}...<extra></extra>"),
            customdata=[youtube_link(h["video_id"], h["start"])],
            showlegend=False,
        ))
    fig.update_layout(
        barmode="overlay", height=120 + 46 * len(titles), margin=dict(l=10, r=10, t=30, b=40),
        xaxis_title="Position in video (minutes)", yaxis=dict(categoryorder="array", categoryarray=titles[::-1]),
        title="Where the best segments are",
    )
    return fig
