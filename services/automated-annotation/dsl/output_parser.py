from typing import Dict, Any


def load_property_id_mapping_api(context_data: Dict[str, Any]) -> Dict[int, str]:
    """Build an ID->name map from API-loaded context_data."""
    id_mapping: Dict[int, str] = {}

    if isinstance(context_data, dict) and 'data' in context_data:
        context_data = context_data['data']

    if isinstance(context_data, dict) and 'context_content' in context_data:
        content = context_data['context_content']
        for item_id, item_data in content.items():
            try:
                if isinstance(item_data, dict):
                    name = item_data.get('name', f'ID {item_id}')
                    id_mapping[int(item_id)] = name
            except (ValueError, TypeError):
                continue
    else:
        items = context_data.get("materials", context_data.get("items", [])) if isinstance(context_data, dict) else []
        for item in items:
            if isinstance(item, str) and item.startswith("ID ") and ":" in item:
                try:
                    id_part = item.split(":")[0].strip()
                    id_num = int(id_part.split()[1])
                    name = item.split(":", 1)[1].strip()
                    id_mapping[id_num] = name
                except (ValueError, IndexError):
                    continue

    return id_mapping
