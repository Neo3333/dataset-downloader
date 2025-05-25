import argparse
from distutils.util import strtobool

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
      "--dest_suffix",
      help="Suffix of the destination directory to save dataset",
      default=''
  )
  parser.add_argument(
    "--parquet_only",
    help="Whether to download parquet format files only",
    type=lambda x: bool(strtobool(x)),
    default=False
)
  args = parser.parse_args()

  download_dataset(
    repo_id=args.dataset,
    config=args.config,
    split=args.split,
    dest_suffix=args.dest_suffix,
    parquet_only=args.parquet_only
  )


if __name__ == "__main__":
  main()