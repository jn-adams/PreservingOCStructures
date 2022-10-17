import networkx as nx


def convert_to_nx_graphs(g,ocel, k, target, from_start=False):


    return_graphs = []
    target_values = []

    ts_pairs = [(idx.event_id, ocel.get_value(idx.event_id, "event_timestamp")) for idx in g.nodes]
    ts_pairs.sort(key=lambda x: x[1])
    sorted_idxs = [p[0] for p in ts_pairs]
    end_index = 0 if from_start else len(sorted_idxs) - k

    #to networkx graph
    nx_graph = nx.Graph()
    g.edges
    for edge in g.edges:
        nx_graph.add_edge(edge.source, edge.target)
    nx.set_node_attributes(nx_graph,{n.event_id:n.attributes for n in g.nodes})

    #extract subgraphs
    for start in range(0, end_index + 1):
        subgraph = nx.subgraph(nx_graph, sorted_idxs[start:start + k]).copy()
        for node in subgraph.nodes():
            val = subgraph.nodes()[node][target]
            del subgraph.nodes()[node][target]

            if node == sorted_idxs[start+k-1]:
                target_values.append(val)


        indexed_subgraph = nx.convert_node_labels_to_integers(subgraph)

        return_graphs.append(indexed_subgraph)

    return return_graphs, target_values



def embed(train_nx_feature_graphs,test_nx_feature_graphs, technique, size= 20):
    if technique == 'FEATHER-G':
        # FEATHER-G from Rozemberczki et al.: Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models (CIKM 2020)
        from karateclub import FeatherGraph
        model = FeatherGraph()
        model.fit(train_nx_feature_graphs+test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'Graph2Vec':
        # Graph2Vec from Narayanan et al.: Graph2Vec: Learning Distributed Representations of Graphs (MLGWorkshop 2017)
        from karateclub import Graph2Vec
        # dimensions: int = 128
        model = Graph2Vec(dimensions = size)
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'NetLSD':
        # NetLSD from Tsitsulin et al.: NetLSD: Hearing the Shape of a Graph (KDD 2018)
        from karateclub import NetLSD
        # scale_steps: int = 250
        model = NetLSD(scale_steps = size)
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'WaveletCharacteristic':
        # WaveletCharacteristic from Wang et al.: Graph Embedding via Diffusion-Wavelets-Based Node Feature Distribution Characterization (CIKM 2021)
        from karateclub import WaveletCharacteristic
        # ?
        model = WaveletCharacteristic()
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'IGE':
        # IGE from Galland et al.: Invariant Embedding for Graph Classification (ICML 2019 LRGSD Workshop)
        from karateclub import IGE
        # ?
        model = IGE()
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'LDP':
        # LDP from Cai et al.: A Simple Yet Effective Baseline for Non-Attributed Graph Classification (ICLR 2019)
        from karateclub import LDP
        # ?
        model = LDP()
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'GeoScattering':
        # GeoScattering from Gao et al.: Geometric Scattering for Graph Data Analysis (ICML 2019)
        from karateclub import GeoScattering
        # ?
        model = GeoScattering()
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'GL2Vec':
        # GL2Vec from Chen and Koga: GL2Vec: Graph Embedding Enriched by Line Graphs with Edge Features (ICONIP 2019)
        from karateclub import GL2Vec
        # dimensions: int = 128
        model = GL2Vec(dimensions = size)
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'SF':
        # SF from de Lara and Pineau: A Simple Baseline Algorithm for Graph Classification (NeurIPS RRL Workshop 2018)
        from karateclub import SF
        # dimensions: int = 128
        model = SF(dimensions = size)
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    elif technique == 'FGSD':
        # FGSD from Verma and Zhang: Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs (NeurIPS 2017)
        from karateclub import FGSD
        # hist_bins: int = 200
        model = FGSD(hist_bins = size)
        model.fit(train_nx_feature_graphs + test_nx_feature_graphs)
        X = model.get_embedding()
        X_train = X[:len(train_nx_feature_graphs)]
        X_test = X[len(train_nx_feature_graphs):]

    else:
        raise AttributeError(f'{technique} does not exist.')

    return X_train, X_test


def test_graph_embedding(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_hat)
    print('AUC: {:.4f}'.format(auc))
