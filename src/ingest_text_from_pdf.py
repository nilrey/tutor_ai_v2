"""
обработка входящего PDF, этап 1 - "выгрузка сырых данных"
"""
import sys
from pathlib import Path
import argparse


def load_pdf():
    return True


def save_txt():
    return True

def get_arg(arg_nmb):
    if len(sys.argv) > arg_nmb:
        print(env_name)
        value = sys.argv[arg_nmb]

    # if value is None:
    #     value = DEFAULT_PATH
    #     print(f'Warn: os.getenv({env_name}) not defined, sys.argv.{arg_nmb} not defined.')
    return value

if __name__ == "__main__":
    print((sys.argv))
    parser = argparse.ArgumentParser( )
    parser.add_argument('--input' )
    parser.add_argument('--PDF_PATH'.lower() )
    parser.add_argument('--PDF_NAME' )
    args = parser.parse_args()
    print("Input:", args.input)
    print("--pdf_path:", args.pdf_path)

    # pdf_path = get_arg('PDF_PATH', 1)
    # pdf_name = get_arg('PDF_NAME', 2)
    # pdf_file = Path(pdf_path).joinpath(pdf_name)
    # print(pdf_file)
