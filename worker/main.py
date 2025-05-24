import argparse

from hf_downloader import download_dataset

def main():
  parser = argparse.ArgumentParser(
      description="HuggingFace Dataset Download Worker"
  )
  parser.add_argument(
      "--dataset", required=True,
      help="HuggingFace dataset repo ID (e.g. user/dataset)"
  )
  parser.add_argument(
      "--config", help="Dataset config name (if applicable)"
  )
  parser.add_argument(
      "--split", help="Dataset split (e.g. train, test)"
  )
  parser.add_argument(
      "--destination", help="Destination directory to save dataset"
  )
  args = parser.parse_args()

  download_dataset(
    repo_id=args.dataset,
    config=args.config,
    split=args.split,
    dest=args.destination
  )


if __name__ == "__main__":
  main()