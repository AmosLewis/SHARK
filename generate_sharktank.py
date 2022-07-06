# Lint as: python3
"""SHARK Tank"""
# python generate_sharktank.py, you have to give a csv tile with [model_name, model_download_url]
# will generate local shark tank folder like this:
#   /SHARK
#     /gen_shark_tank
#       /tflite
#         /albert_lite_base
#         /...model_name...
#       /tf
#       /pytorch
#

import os
import urllib.request
import csv
import argparse
from shark.shark_importer import SharkImporter


class SharkTank:
    def __init__(
        self,
        torch_model_list: str = None,
        tf_model_list: str = None,
        tflite_model_list: str = None,
        upload: bool = False,
    ):
        self.torch_model_list = torch_model_list
        self.tf_model_list = tf_model_list
        self.tflite_model_list = tflite_model_list
        self.upload = upload

        print("Setting up for TMP_DIR")
        self.workdir = os.path.join(os.path.dirname(__file__), "./gen_shark_tank")
        print(f"tflite TMP_shark_tank_DIR = {self.workdir}")
        os.makedirs(self.workdir, exist_ok=True)

        if self.torch_model_list is not None:
            self.save_torch_model()

        if self.tf_model_list is not None:
            self.save_tf_model()

        print("self.tflite_model_list: ", self.tflite_model_list)
        # compile and run tfhub tflite
        if self.tflite_model_list is not None:
            self.save_tflite_model()

        if self.upload:
            print("upload tmp tank to gcp")
            os.system("gsutil cp -r ./gen_shark_tank gs://shark_tank/")

    def save_torch_model(self):
        from tank.model_utils import get_hf_model

        print("Torch sharktank not implemented yet")

    def save_tf_model(self):
        print("tf sharktank not implemented yet")

    def save_tflite_model(self):
        from shark.tflite_utils import TFLitePreprocessor

        with open(self.tflite_model_list) as csvfile:
            tflite_reader = csv.reader(csvfile, delimiter=",")
            for row in tflite_reader:
                tflite_model_name = row[0]
                tflite_model_link = row[1]
                print("tflite_model_name", tflite_model_name)
                print("tflite_model_link", tflite_model_link)
                tflite_model_name_dir = os.path.join(self.workdir, str(tflite_model_name))
                os.makedirs(tflite_model_name_dir, exist_ok=True)
                print(f"TMP_TFLITE_MODELNAME_DIR = {tflite_model_name_dir}")

                tflite_tosa_file = "/".join(
                    [
                        tflite_model_name_dir,
                        str(tflite_model_name) + "_tflite.mlir",
                    ]
                )

                # Preprocess to get SharkImporter input args
                tflite_preprocessor = TFLitePreprocessor(str(tflite_model_name))
                raw_model_file_path = tflite_preprocessor.get_raw_model_file()
                inputs = tflite_preprocessor.get_inputs()
                tflite_interpreter = tflite_preprocessor.get_interpreter()

                # Use SharkImporter to get SharkInference input args
                my_shark_importer = SharkImporter(
                    module=tflite_interpreter,
                    inputs=inputs,
                    frontend="tflite",
                    raw_model_file=raw_model_file_path,
                )
                mlir_model, func_name = my_shark_importer.import_mlir()

                if os.path.exists(tflite_tosa_file):
                    print("Exists", tflite_tosa_file)
                else:
                    mlir_str = mlir_model.decode("utf-8")
                    with open(tflite_tosa_file, "w") as f:
                        f.write(mlir_str)
                    print(f"Saved mlir in {tflite_tosa_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch_model_list",
        type=str,
        default="./tank/torch/torch_model_list.csv",
    )
    parser.add_argument("--tf_model_list", type=str, default="./tank/tf/tf_model_list.csv")
    parser.add_argument(
        "--tflite_model_list",
        type=str,
        default="./tank/tflite/tflite_model_list.csv",
    )
    parser.add_argument("--upload", type=bool, default=False)
    args = parser.parse_args()
    SharkTank(
        torch_model_list=args.torch_model_list,
        tf_model_list=args.tf_model_list,
        tflite_model_list=args.tflite_model_list,
        upload=args.upload,
    )
