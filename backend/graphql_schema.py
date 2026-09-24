"""GraphQL schema for the Nectar asset-connectivity bonus.

The schema exposes the four connectivity queries required by the
Nectar challenge and uses the same saved Task 5 graph/data as Flask.
"""

from graphql import (
    GraphQLArgument,
    GraphQLField,
    GraphQLFloat,
    GraphQLInt,
    GraphQLList,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
)


AssetConnectionType = GraphQLObjectType(
    name="AssetConnection",
    fields=lambda: {
        "asset_id": GraphQLString,
        "direction": GraphQLString,
        "connection_type": GraphQLString,
        "relationship_strength": GraphQLFloat,
    },
)


DownstreamAssetType = GraphQLObjectType(
    name="DownstreamAsset",
    fields=lambda: {
        "asset_id": GraphQLString,
        "depth": GraphQLInt,
        "connection_type": GraphQLString,
        "relationship_strength": GraphQLFloat,
    },
)


AssetType = GraphQLObjectType(
    name="Asset",
    fields=lambda: {
        "asset_id": GraphQLString,
        "asset_name": GraphQLString,
        "asset_type": GraphQLString,
        "site_id": GraphQLString,
        "building_id": GraphQLString,
    },
)


def _metadata_asset_rows(metadata):
    """Convert metadata rows to GraphQL-safe dictionaries."""
    records = []

    for row in metadata.to_dict("records"):
        records.append(
            {
                "asset_id": (
                    None
                    if row.get("asset_id") is None
                    else str(row.get("asset_id"))
                ),
                "asset_name": (
                    None
                    if row.get("asset_name") is None
                    else str(row.get("asset_name"))
                ),
                "asset_type": (
                    None
                    if row.get("asset_type") is None
                    else str(row.get("asset_type"))
                ),
                "site_id": (
                    None
                    if row.get("site_id") is None
                    else str(row.get("site_id"))
                ),
                "building_id": (
                    None
                    if row.get("building_id") is None
                    else str(row.get("building_id"))
                ),
            }
        )

    return records


def _connection_rows(rows):
    """Convert graph-service output to GraphQL-safe dictionaries."""
    result = []

    for row in rows:
        result.append(
            {
                "asset_id": (
                    None
                    if row.get("asset_id") is None
                    else str(row.get("asset_id"))
                ),
                "direction": (
                    None
                    if row.get("direction") is None
                    else str(row.get("direction"))
                ),
                "connection_type": (
                    None
                    if row.get("connection_type") is None
                    else str(row.get("connection_type"))
                ),
                "relationship_strength": (
                    None
                    if row.get("relationship_strength") is None
                    else float(row.get("relationship_strength"))
                ),
            }
        )

    return result


def _downstream_rows(rows):
    """Convert downstream output to GraphQL-safe dictionaries."""
    result = []

    for row in rows:
        result.append(
            {
                "asset_id": (
                    None
                    if row.get("asset_id") is None
                    else str(row.get("asset_id"))
                ),
                "depth": (
                    None
                    if row.get("depth") is None
                    else int(row.get("depth"))
                ),
                "connection_type": (
                    None
                    if row.get("connection_type") is None
                    else str(row.get("connection_type"))
                ),
                "relationship_strength": (
                    None
                    if row.get("relationship_strength") is None
                    else float(row.get("relationship_strength"))
                ),
            }
        )

    return result


def create_schema(graph, metadata):
    """Create GraphQL schema around the existing Task 5 graph."""

    def resolve_connected_assets(_root, info, assetName=None):
        services = info.context["services"]

        if not assetName:
            return []

        asset_df = metadata[
            metadata["asset_name"].astype(str)
            == str(assetName)
        ]

        if asset_df.empty:
            return []

        asset_id = asset_df.iloc[0]["asset_id"]

        rows = services["connected_assets"](
            graph,
            asset_id,
        )

        return _connection_rows(rows)

    def resolve_downstream_assets(_root, info, assetName=None):
        services = info.context["services"]

        if not assetName:
            return []

        asset_df = metadata[
            metadata["asset_name"].astype(str)
            == str(assetName)
        ]

        if asset_df.empty:
            return []

        asset_id = asset_df.iloc[0]["asset_id"]

        rows = services["downstream_assets"](
            graph,
            asset_id,
        )

        return _downstream_rows(rows)

    def resolve_assets_under_site(_root, info, siteId=None):
        services = info.context["services"]

        if not siteId:
            return []

        result = services["assets_under_site"](
            metadata,
            siteId,
        )

        return _metadata_asset_rows(result)

    def resolve_isolated_assets(_root, info):
        services = info.context["services"]

        result = services["isolated_assets"](
            graph,
            metadata,
        )

        return _metadata_asset_rows(result)

    QueryType = GraphQLObjectType(
        name="Query",
        fields={
            "connectedAssets": GraphQLField(
                GraphQLList(AssetConnectionType),
                args={
                    "assetName": GraphQLArgument(
                        GraphQLString
                    )
                },
                resolve=resolve_connected_assets,
            ),
            "downstreamAssets": GraphQLField(
                GraphQLList(DownstreamAssetType),
                args={
                    "assetName": GraphQLArgument(
                        GraphQLString
                    )
                },
                resolve=resolve_downstream_assets,
            ),
            "assetsUnderSite": GraphQLField(
                GraphQLList(AssetType),
                args={
                    "siteId": GraphQLArgument(
                        GraphQLString
                    )
                },
                resolve=resolve_assets_under_site,
            ),
            "isolatedAssets": GraphQLField(
                GraphQLList(AssetType),
                resolve=resolve_isolated_assets,
            ),
        },
    )

    return GraphQLSchema(query=QueryType)
