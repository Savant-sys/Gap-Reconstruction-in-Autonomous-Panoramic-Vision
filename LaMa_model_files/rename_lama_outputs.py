import os

# --------------------------------------------------
# Folder containing LaMa outputs
# --------------------------------------------------
DIR = "/mnt/d/edgeconnect_project/lama_outputs_threecam"


def main():
    files = os.listdir(DIR)

    renamed = 0
    skipped = 0

    for fname in files:
        if "_mask001" not in fname:
            skipped += 1
            continue

        old_path = os.path.join(DIR, fname)

        # remove suffix
        new_name = fname.replace("_mask001", "")
        new_path = os.path.join(DIR, new_name)

        # avoid overwriting existing files
        if os.path.exists(new_path):
            print(f"[skip] already exists: {new_name}")
            skipped += 1
            continue

        os.rename(old_path, new_path)
        renamed += 1

        print(f"[renamed] {fname} -> {new_name}")

    print("\nDone.")
    print(f"Renamed: {renamed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()