import sys
from reddwarf.data_loader import Loader

def main():
    if len(sys.argv) == 3:
        convo_id = sys.argv[1]
        directory_name = sys.argv[2]
        filepath = f"tests/fixtures/{directory_name}"
        print(f"Downloading Polis data from conversation {convo_id} into {filepath}...")
        Loader(conversation_id=convo_id, output_dir=filepath)
    else:
        raise ValueError("Must pass a convo_id and directory_name as args.")

if __name__ == "__main__":
    main()
