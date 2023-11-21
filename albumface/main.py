import sys
import argparse
from pathlib import Path
import os
from albumface.utils import album as utils

def main():
    parser = argparse.ArgumentParser("albumface cli for test album image")
    parser.add_argument("-v", "--version", help="version", default=False, action="store_true")

    subparsers = parser.add_subparsers(dest="command")
    #generate albums
    generate_parser = subparsers.add_parser("generate-albums")
    generate_parser.add_argument("-p", "--path", help="path to image file", required=True)
    generate_parser.add_argument("-oa", "--outputalbum", help="json to save album data", required=True)
    generate_parser.add_argument("-oi", "--outputimage", help="json to save image data", required=True)

    #get similiar selfie
    get_similiar = subparsers.add_parser("selfie")
    get_similiar.add_argument("-p", "--path", help="path to image file", required=True)
    get_similiar.add_argument("-o", "--output", help="json file to show similiar image", required=True)
    get_similiar.add_argument("-d", "--data", help="json file that store album data", required=True)
    args = parser.parse_args()

    version =  "0.0.0.0.3"
    if args.version:
        print(version)
        exit(0)

    elif args.command == "generate-albums":
        if args.path and args.outputalbum and args.outputimage:
            try:
                utils.generate_album(path=args.path, album_data_json=args.outputalbum, image_data_json=args.outputimage)
            except Exception as e:
                print("error " + repr(e))
                raise SystemExit(1)

    elif args.command == "selfie":
        if args.path and args.output and args.data:
            try:
                data = utils.generate_face_embeddings(path=args.path)
                if len(data) != 1:
                    print("error only allowed one face")
                    raise SystemExit(1)
                else:
                    utils.get_selfie_response(person_embd=data[0], album_data_json=args.data, selfie_data_json=args.output)
            except Exception as e:
                print("error " + repr(e))
                raise SystemExit(1)



    if __name__ == "__main__":
        main()