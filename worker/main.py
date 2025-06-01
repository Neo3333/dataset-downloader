import argparse
from distutils.util import strtobool

from hf_downloader import download_huggingface_dataset
from kaggle_downloader import download_kaggle_dataset

def main():
  parser = argparse.ArgumentParser(
    description="Third Party Dataset Download Worker"
  )
  parser.add_argument(
    "--source", choices=["huggingface", "kaggle"], required=True,
    help="Data source: 'huggingface' or 'kaggle'"
  )
  parser.add_argument(
    "--dataset", required=True,
    help="Dataset repo ID (e.g. user/dataset)"
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

  if args.source == 'huggingface':
    download_huggingface_dataset(
      repo_id=args.dataset,
      config=args.config,
      split=args.split,
      dest_suffix=args.dest_suffix,
      parquet_only=args.parquet_only
    )
  elif args.source == 'kaggle':
    download_kaggle_dataset(
      repo_id=args.dataset,
      dest_suffix=args.dest_suffix
    )
  else:
    raise ValueError(f"Unknown source '{args.source}'")


if __name__ == "__main__":
  main()