# Examples

This repository contains example inputs for single and batch processing across multiple data formats: TXT, JSON, CSV, and images.

All examples assume the current directory is bind-mounted when running containers or scripts.

## Text / JSON / CSV Examples (Batch & Single)

Clinical-style data originates from the Primock57 dataset:

https://github.com/babylonhealth/primock57/tree/main

The data in this repository was generated and reformatted using the following Python script, which:

Groups consultation notes by patient ID

Exports per-patient data as:

- JSON
- TXT (human-readable)
- CSV (flattened)

Data export script:

    import json
    import os
    import glob
    import re
    import pandas as pd

    NOTES_DIR = "notes"
    OUT_DIR = "output"

    os.makedirs(OUT_DIR, exist_ok=True)

    def extract_patient_id(filename):
        """
        Extract patient ID from filename.
        Example: day1_consultation01.json -> 01
        """
        match = re.search(r'consultation(\d+)\.json$', filename)
        if not match:
            raise ValueError(f"Cannot extract patient ID from {filename}")
        return match.group(1)

    # Group notes by patient ID
    patients = {}

    for path in glob.glob(os.path.join(NOTES_DIR, "*.json")):
        filename = os.path.basename(path)
        patient_id = extract_patient_id(filename)

        with open(path, "r", encoding="utf-8") as f:
            note = json.load(f)

        note["_source_file"] = filename
        patients.setdefault(patient_id, []).append(note)

    print(f"Found {len(patients)} patients")
    # Write output per patient
    for patient_id, notes in patients.items():
        patient_dir = os.path.join(OUT_DIR, f"patient_{patient_id}")
        os.makedirs(patient_dir, exist_ok=True)

        # JSON
        with open(os.path.join(patient_dir, f"patient_{patient_id}.json"), "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=4, ensure_ascii=False)

        # TXT
        def write_txt_vertical(notes, out_path):
            with open(out_path, "w", encoding="utf-8") as f:
                for i, note in enumerate(notes, start=1):
                    f.write(f"--- Consultation {i} ---\n")
                    for key, value in note.items():
                        if isinstance(value, (dict, list)):
                            f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")

        txt_path = os.path.join(patient_dir, f"patient_{patient_id}.txt")
        write_txt_vertical(notes, txt_path)

        # CSV (flattened)
        try:
            df = pd.json_normalize(notes)
            df.to_csv(
                os.path.join(patient_dir, f"patient_{patient_id}.csv"),
                index=False,
                encoding="utf-8"
            )
        except Exception as e:
            print(f"CSV failed for patient {patient_id}: {e}")

        print(f"Exported patient {patient_id}")

## Image Examples

### Batch Image

Chest X-ray datasets are sourced from Hugging Face:

https://huggingface.co/datasets/UniqueData/chest-x-rays

### Single Image

The single-image example uses a chest X-ray:

Author: Stillwaterising

License: CC0 (Public Domain)

Source: Wikimedia Commons

https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png