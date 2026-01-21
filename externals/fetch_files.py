import argparse
import os
import sys
import urllib.request

DOWNLOAD_DICT = {
  "antlr.jar": ["https://www.antlr.org/download/antlr-4.13.2-complete.jar"],
}

def try_download_file(msg_prefix, filename, urls):
  abspath = os.path.join(os.path.dirname(__file__), filename)
  if os.path.exists(abspath):
    print(f"{msg_prefix} {filename} exists, skipping")
    return
  print(f"{msg_prefix} Downloading {filename}")
  for url in urls:
    try:
      urllib.request.urlretrieve(url, abspath)
      print(f"    Downloaded from {url}")
      break
    except:
      print(f"    Failed to download from {url}")

def try_download_all():
  for idx, (filename, urls) in enumerate(DOWNLOAD_DICT.items()):
    msg_prefix = f"[{idx+1}/{len(DOWNLOAD_DICT)}]"
    try_download_file(msg_prefix, filename, urls)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--all", action="store_true", help="Download all files")
  parser.add_argument("filenames", nargs="*", help="Filenames to download")
  args = parser.parse_args()
  if args.all:
    try_download_all()
  else:
    for filename in args.filenames:
      try_download_file("", filename, DOWNLOAD_DICT[filename])
  return 0

if __name__ == "__main__":
  sys.exit(main())
