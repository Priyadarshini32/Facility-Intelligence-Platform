"""Task 5 connectivity service using the saved connectivity data."""

import networkx as nx
import pandas as pd


def build_graph(connectivity):
    """Build the same directed graph used in Task 5."""
    graph = nx.DiGraph()

    for row in connectivity.itertuples(index=False):
        graph.add_edge(
            row.source_asset_id,
            row.target_asset_id,
            connection_type=row.connection_type,
            relationship_strength=float(
                row.relationship_strength
            ),
        )

    return graph


def load_saved_connectivity(path):
    """Load the connectivity relationships saved by Task 5."""
    return pd.read_csv(path)


def connected_assets(graph, asset_id):
    """Return direct incoming and outgoing relationships."""

    if asset_id not in graph:
        return []

    results = []

    for source, target, data in graph.edges(
        asset_id,
        data=True,
    ):
        results.append(
            {
                "asset_id": target,
                "direction": "outgoing",
                "connection_type": data.get(
                    "connection_type"
                ),
                "relationship_strength": data.get(
                    "relationship_strength"
                ),
            }
        )

    for source, target, data in graph.in_edges(
        asset_id,
        data=True,
    ):
        results.append(
            {
                "asset_id": source,
                "direction": "incoming",
                "connection_type": data.get(
                    "connection_type"
                ),
                "relationship_strength": data.get(
                    "relationship_strength"
                ),
            }
        )

    return results


def downstream_assets(graph, asset_id):
    """Return directed downstream assets."""

    if asset_id not in graph:
        return []

    results = []

    for node in nx.descendants(
        graph,
        asset_id,
    ):
        path = nx.shortest_path(
            graph,
            asset_id,
            node,
        )

        edge_data = graph.get_edge_data(
            path[-2],
            node,
            default={},
        )

        results.append(
            {
                "asset_id": node,
                "depth": len(path) - 1,
                "connection_type": edge_data.get(
                    "connection_type"
                ),
                "relationship_strength": edge_data.get(
                    "relationship_strength"
                ),
            }
        )

    return sorted(
        results,
        key=lambda x: (
            x["depth"],
            x["asset_id"],
        ),
    )


def assets_under_site(metadata, site_id):
    return (
        metadata[
            metadata["site_id"].astype(str)
            == str(site_id)
        ]
        .copy()
        .reset_index(drop=True)
    )


def isolated_assets(graph, metadata):
    """Assets with no incoming or outgoing connectivity."""
    connected_ids = set(graph.nodes)

    return (
        metadata[
            ~metadata["asset_id"].isin(
                connected_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )
