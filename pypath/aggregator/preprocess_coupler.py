'''WMPC 保留的实例解析与局部节点缓存接口。

这里只保留稀疏 Schwarz 需要的三个函数，训练期的聚合器流水线不迁入 WMPC。
'''
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from pypath.aggregator.cell_pool_utils import get_aggregator_cell_pool


def parse_netlist_for_instances(filepath: str) -> Dict[str, Any]:
    instances: Dict[str, Any] = {}
    cell_pool = get_aggregator_cell_pool()
    try:
        with open(filepath, 'r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip().upper()
                if not line.startswith('X'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                instance_name = parts[0]
                cell_type = parts[-1]
                connections = parts[1:-1]
                if cell_type in cell_pool:
                    instances[instance_name.lower()] = {
                        'type': cell_type,
                        'connections': connections,
                    }
    except FileNotFoundError:
        return {}
    return instances


def extract_uut_only_features(
    node_map: Dict[str, int],
    uut_instance_name: str,
    instance_info: Dict[str, Any],
) -> Tuple[List[int] | None, List[str] | None, List[str] | None, Dict[str, str] | None]:
    instance_name = str(uut_instance_name).lower()
    cell_type = str(instance_info.get('type', ''))
    cell_info = get_aggregator_cell_pool().get(cell_type)
    if cell_info is None:
        return None, None, None, None

    connections = list(instance_info.get('connections', []))
    external_pin_to_global_node = {
        str(pin): str(node)
        for pin, node in zip(cell_info.get('pins', []), connections)
        if str(node).upper() not in {'VDD', 'VSS'}
    }
    canonical_external_pins = [
        str(pin)
        for pin in cell_info.get('pins', [])
        if str(pin).upper() not in {'VDD', 'VSS'}
    ]
    external_nodes = {value.lower() for value in external_pin_to_global_node.values()}
    internal_nodes: List[str] = []
    start_pattern = f'{instance_name}.'
    search_pattern = f'.{instance_name}.'
    for node_name in node_map:
        lowered = str(node_name).lower()
        if not (lowered.startswith(start_pattern) or search_pattern in lowered):
            continue
        if lowered in external_nodes:
            continue
        internal_nodes.append(str(node_name))
    local_names = canonical_external_pins + sorted(internal_nodes)

    indices: List[int] = []
    valid_names: List[str] = []
    for local_name in local_names:
        global_name = external_pin_to_global_node.get(local_name, local_name)
        matrix_index = node_map.get(str(global_name).lower())
        if matrix_index is None:
            continue
        indices.append(int(matrix_index) - 1)
        valid_names.append(local_name)
    if not indices:
        return None, None, None, None
    return indices, local_names, valid_names, external_pin_to_global_node


def build_instance_feature_cache(
    node_map: Dict[str, int],
    instances: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    for instance_name, instance_info in instances.items():
        indices, local_names, valid_names, external_map = extract_uut_only_features(
            node_map, instance_name, instance_info
        )
        if indices is None:
            continue
        cache[str(instance_name)] = {
            'cell_type': str(instance_info.get('type', '')),
            'uut_indices': np.asarray(indices, dtype=np.int64),
            'uut_local_node_names': tuple(valid_names or ()),
            'canonical_uut_local_node_names': tuple(local_names or ()),
            'external_pin_to_global_node': dict(external_map or {}),
        }

    global_to_pin_refs: Dict[str, List[Tuple[str, str, str]]] = {}
    for instance_name, item in cache.items():
        for pin_name, global_node in item['external_pin_to_global_node'].items():
            global_to_pin_refs.setdefault(str(global_node).lower(), []).append(
                (instance_name, str(pin_name), str(item['cell_type']))
            )
    for instance_name, item in cache.items():
        token_specs: List[Dict[str, str]] = []
        for pin_name, global_node in item['external_pin_to_global_node'].items():
            for other_name, other_pin, other_type in global_to_pin_refs.get(
                str(global_node).lower(), []
            ):
                if other_name == instance_name:
                    continue
                token_specs.append(
                    {
                        'focal_pin': str(pin_name),
                        'neighbor_instance': other_name,
                        'neighbor_pin': other_pin,
                        'neighbor_cell_type': other_type,
                        'shared_global_node': str(global_node),
                    }
                )
        item['neighbor_token_specs'] = sorted(
            token_specs,
            key=lambda value: (
                value['focal_pin'],
                value['neighbor_instance'],
                value['neighbor_pin'],
                value['neighbor_cell_type'],
                value['shared_global_node'],
            ),
        )
    return cache
