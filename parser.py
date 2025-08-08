import os
import json
import re
import unicodedata
import xml.etree.ElementTree as ET

# ---------- CONFIG ----------
XML_DIR = "./ecospold_xml/EcoSpold01"  # Folder with XML files
OUTPUT_DIR = "./processed_json"        # Output folder for JSON
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- HELPERS ----------
def slugify(value):
    """Convert string to safe filename."""
    value = unicodedata.normalize('NFKD', value)
    value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^a-zA-Z0-9]+', '-', value)
    return value.strip('-').lower()

def parse_xml(file_path):
    """Parse EcoSpold XML and return structured process data."""
    ns = {"eco": "http://www.EcoInvent.org/EcoSpold01"}
    tree = ET.parse(file_path)
    root = tree.getroot()

    # --- Reference function (process metadata)
    ref_func = root.find(".//eco:referenceFunction", ns)
    if ref_func is None:
        return None

    process_name = ref_func.attrib.get("name", "unknown_process").strip()
    description = ref_func.attrib.get("generalComment", "").strip()

    category = ref_func.attrib.get("category", "").strip()
    subcategory = ref_func.attrib.get("subCategory", "").strip()

    # --- Exchanges (inputs & outputs)
    inputs, outputs = [], []

    for exch in root.findall(".//eco:exchange", ns):
        exch_data = {
            "name": exch.attrib.get("name", "").strip(),
            "category": exch.attrib.get("category", "").strip(),
            "subcategory": exch.attrib.get("subCategory", "").strip(),
            "amount": float(exch.attrib.get("meanValue", "0") or 0),
            "unit": exch.attrib.get("unit", "").strip(),
            "description": exch.attrib.get("generalComment", "").strip()
        }
        if exch.find("eco:inputGroup", ns) is not None:
            inputs.append(exch_data)
        elif exch.find("eco:outputGroup", ns) is not None:
            outputs.append(exch_data)

    # --- Prepare structured JSON for RAG
    process_data = {
        "id": slugify(process_name),
        "name": process_name,
        "description": description if description else "No description provided.",
        "category": category,
        "subcategory": subcategory,
        "inputs": inputs,
        "outputs": outputs
    }

    # --- Add combined field for embedding
    # This can be used to generate embeddings without losing structure
    embedding_text = (
        f"{process_name}. "
        f"{description if description else 'No description provided.'} "
        f"Category: {category} / {subcategory}. "
        f"Inputs: " + (", ".join(
            f"{i['name']} ({i['amount']} {i['unit']}) - {i['description']}" if i['description'] else f"{i['name']} ({i['amount']} {i['unit']})"
            for i in inputs
        ) if inputs else "None") + ". "
        f"Outputs: " + (", ".join(
            f"{o['name']} ({o['amount']} {o['unit']}) - {o['description']}" if o['description'] else f"{o['name']} ({o['amount']} {o['unit']})"
            for o in outputs
        ) if outputs else "None")
    )

    process_data["text_for_embedding"] = embedding_text

    return process_data

# ---------- MAIN ----------
def main():
    for file_name in os.listdir(XML_DIR):
        if not file_name.lower().endswith(".xml"):
            continue

        file_path = os.path.join(XML_DIR, file_name)
        process_data = parse_xml(file_path)
        if not process_data:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{process_data['id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(process_data, f, indent=2, ensure_ascii=False)

        print(f"Processed: {file_name} → {out_path}")

if __name__ == "__main__":
    main()
