import os


def rename_recursive(root_dir: str, key: str, replacement: str, dry_run: bool = False):
    # Walk bottom-up so directories are renamed after their contents
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # print(f"Directory: {dirpath}")
        # for dirname in dirnames:
        #     print(f"  Subdirectory: {dirname}")
        # for filename in filenames:
        #     print(f"  File: {filename}")
        # Rename files
        # print(filenames)
        for name in filenames:
            if key in name:
                old_path = os.path.join(dirpath, name)
                new_name = name.replace(key, replacement)
                new_path = os.path.join(dirpath, new_name)
                print(f"Rename file: {old_path} -> {new_path}")
                if not dry_run:
                    os.rename(old_path, new_path)

        # Rename directories
        for name in dirnames:
            if key in name:
                old_path = os.path.join(dirpath, name)
                new_name = name.replace(key, replacement)
                new_path = os.path.join(dirpath, new_name)
                print(f"Rename dir:  {old_path} -> {new_path}")
                if not dry_run:
                    os.rename(old_path, new_path)


if __name__ == "__main__":
    # Example usage:
    # Go through all folders under "./project",
    # and replace "oldkey" with "newkey" in names.
    root_directory = "/home/jonathan/catkin_ws/src/frasa/logs/crossq"
    key_to_find = "frasa-standup-v0"
    replacement = "unified-humanoid-get-up-env-standup-v0"

    rename_recursive(root_directory, key_to_find, replacement, False)
