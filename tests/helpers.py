from reddwarf.types.polis import PolisRepness
from typing import Union

def get_grouped_statement_ids(repness: PolisRepness) -> dict[str, list[dict[str, list[int]]]]:
    """A helper to compare only tid in groups, rather than full repness object."""
    groups = []

    for key, statements in repness.items():
        group = {"id": str(key), "members": sorted([stmt["tid"] for stmt in statements])} # type:ignore
        groups.append(group)

    return {"groups": groups}

def pad_to_size(lst, size):
    return list(lst) + [[0., 0.]]*(size - len(lst))

def transform_base_clusters_to_participant_coords(base_clusters):
    """
    Transform base clusters data into a list of dictionaries with participant_id and xy coordinates.

    Args:
        base_clusters (dict): A dictionary containing base clusters data with 'id', 'members', 'x', and 'y' keys.

    Returns:
        list: A list of dictionaries, each containing a participant_id and their xy coordinates.
    """
    # For now, ensure failure if a base-cluster has more than one member, as the test assumes that.
    def get_only_member_or_raise(members):
        if len(members) != 1:
            raise Exception("A base-cluster has more than one member when it cannot")

        return members[0]

    return [
        {
            "participant_id": get_only_member_or_raise(members),
            "xy": [x, y]
        }
        for members, x, y in zip(
            base_clusters['members'],
            base_clusters['x'],
            base_clusters['y']
        )
    ]

# Not used right now. Maybe later.
def groupsort_pids_by_cluster(df):
    """
    Helper function to gather statement IDS in clusters and sort them for easy
    comparison.

    This make comparison easy, even when kmeans gives different numeric labels.

    Args:
        df (pd.DataFrame): A dataframe with projected participants, columns "x",
        "y", "cluster_id"
    Returns:
        (list[list[int]]): A list of lists, each containing statement IDs in a
        cluster.
    """
    # Group by cluster_id and collect indices
    grouped = df.groupby('cluster_id').apply(lambda x: list(x.index))

    # Sort the groups by their length (number of members) in descending order
    sorted_groups = sorted(grouped, key=len, reverse=True)

    # Convert each inner list to integers
    return [list(map(int, group)) for group in sorted_groups]


NestedValue = Union[int, float, list["NestedValue"], dict[str, "NestedValue"]]
NestedDict = dict[str, NestedValue]

def flip_signs_by_key(nested_dict: NestedDict, keys: list[str] = []) -> NestedDict:
    """
    Flips the signs of numeric values in a nested dict using dot-notation paths.
    Supports nested arrays and array indexing like "foo.bar[0].baz[1]".

    This helper is for quickly dealing with real polismath fixture data that has
    different signs than we calculate ourselves.

    NOTE: The need for this helper may be harmless artifacts of PCA methods, or
    other reasons related to agree/disagree signs. Consistently adjusting signs
    at the fixture level should help clarify this.

    Arguments:
        obj (NestedDict): A nested dict of arbitrary depth, with lists of float values at some keys.
        keys (list[str]): A list of dot-notation keys to flip signs within.

    Returns:
        NestedDict: A nested dict with all the same original types, but with specific keys inverted.
    """
    import copy
    import re
    result: NestedDict = copy.deepcopy(nested_dict)

    def flip_recursive(value: NestedValue) -> NestedValue:
        if isinstance(value, list):
            return [flip_recursive(v) for v in value]
        elif isinstance(value, (int, float)):
            return -value
        return value

    def parse_path_segment(segment: str) -> list[Union[str, int]]:
        parts: list[Union[str, int]] = []
        for match in re.finditer(r'([^\[\]]+)|\[(\d+)\]', segment):
            key, idx = match.groups()
            if key:
                parts.append(key)
            elif idx:
                parts.append(int(idx))
        return parts

    def get_parent_and_final_key(root: NestedValue, path: str) -> tuple[Union[dict, list, None], Union[str, int, None]]:
        segments = path.split(".")
        parts: list[Union[str, int]] = []
        for seg in segments:
            parts.extend(parse_path_segment(seg))

        current = root
        for part in parts[:-1]:
            if isinstance(part, str) and isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(part, int) and isinstance(current, list) and 0 <= part < len(current):
                current = current[part]
            else:
                return None, None
        return current, parts[-1] if parts else None

    for dot_key in keys:
        parent, final_key = get_parent_and_final_key(result, dot_key)
        if parent is None or final_key is None:
            continue

        if isinstance(final_key, str) and isinstance(parent, dict) and final_key in parent:
            parent[final_key] = flip_recursive(parent[final_key])
        elif isinstance(final_key, int) and isinstance(parent, list) and 0 <= final_key < len(parent):
            parent[final_key] = flip_recursive(parent[final_key])

    return result