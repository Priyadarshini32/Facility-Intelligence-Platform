"""GraphQL schema for Nectar asset connectivity."""

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


# ------------------------------------------------------------
# GraphQL Types
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _clean(value):
    """Normalize IDs/names for comparison."""
    if value is None:
        return ""

    return str(value).strip().lower()


def _metadata_asset_rows(metadata):
    """Convert metadata rows to GraphQL-safe dictionaries."""

    records = []

    for row in metadata.to_dict("records"):
        records.append(
            {
                "asset_id": (
                    None
                    if pd_is_null(row.get("asset_id"))
                    else str(row.get("asset_id"))
                ),
                "asset_name": (
                    None
                    if pd_is_null(row.get("asset_name"))
                    else str(row.get("asset_name"))
                ),
                "asset_type": (
                    None
                    if pd_is_null(row.get("asset_type"))
                    else str(row.get("asset_type"))
                ),
                "site_id": (
                    None
                    if pd_is_null(row.get("site_id"))
                    else str(row.get("site_id"))
                ),
                "building_id": (
                    None
                    if pd_is_null(row.get("building_id"))
                    else str(row.get("building_id"))
                ),
            }
        )

    return records


def pd_is_null(value):
    """Small helper without requiring pandas import here."""
    try:
        return value != value
    except Exception:
        return value is None


def _connection_rows(rows):
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


# ------------------------------------------------------------
# Schema
# ------------------------------------------------------------

def create_schema(graph, metadata):
    """Create GraphQL schema around the Task 5 graph."""

    def find_asset_id(asset_value):
        """
        Resolve either an asset name or asset ID
        to the actual asset ID used by the graph.
        """

        if not asset_value:
            return None

        value = _clean(asset_value)

        # First try asset_name
        for _, row in metadata.iterrows():

            if _clean(row.get("asset_name")) == value:
                return str(row["asset_id"]).strip()

        # Then allow asset_id as fallback
        for _, row in metadata.iterrows():

            if _clean(row.get("asset_id")) == value:
                return str(row["asset_id"]).strip()

        return None


    # --------------------------------------------------------
    # Connected Assets
    # --------------------------------------------------------

    def resolve_connected_assets(
        _root,
        info,
        assetName=None,
    ):

        services = info.context["services"]

        asset_id = find_asset_id(assetName)

        if asset_id is None:
            return []

        rows = services["connected_assets"](
            graph,
            asset_id,
        )

        return _connection_rows(rows)


    # --------------------------------------------------------
    # Downstream Assets
    # --------------------------------------------------------

    def resolve_downstream_assets(
        _root,
        info,
        assetName=None,
    ):

        services = info.context["services"]

        asset_id = find_asset_id(assetName)

        if asset_id is None:
            return []

        rows = services["downstream_assets"](
            graph,
            asset_id,
        )

        return _downstream_rows(rows)


    # --------------------------------------------------------
    # Assets Under Site
    # --------------------------------------------------------

    def resolve_assets_under_site(
        _root,
        info,
        siteId=None,
    ):

        services = info.context["services"]

        if not siteId:
            return []

        result = services["assets_under_site"](
            metadata,
            siteId,
        )

        return _metadata_asset_rows(result)


    # --------------------------------------------------------
    # Isolated Assets
    # --------------------------------------------------------

    def resolve_isolated_assets(
        _root,
        info,
    ):

        services = info.context["services"]

        result = services["isolated_assets"](
            graph,
            metadata,
        )

        return _metadata_asset_rows(result)


    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    QueryType = GraphQLObjectType(
        name="Query",

        fields={

            "connectedAssets": GraphQLField(
                GraphQLList(
                    AssetConnectionType
                ),

                args={
                    "assetName": GraphQLArgument(
                        GraphQLString
                    )
                },

                resolve=resolve_connected_assets,
            ),

            "downstreamAssets": GraphQLField(
                GraphQLList(
                    DownstreamAssetType
                ),

                args={
                    "assetName": GraphQLArgument(
                        GraphQLString
                    )
                },

                resolve=resolve_downstream_assets,
            ),

            "assetsUnderSite": GraphQLField(
                GraphQLList(
                    AssetType
                ),

                args={
                    "siteId": GraphQLArgument(
                        GraphQLString
                    )
                },

                resolve=resolve_assets_under_site,
            ),

            "isolatedAssets": GraphQLField(
                GraphQLList(
                    AssetType
                ),

                resolve=resolve_isolated_assets,
            ),
        },
    )

    return GraphQLSchema(
        query=QueryType
    )