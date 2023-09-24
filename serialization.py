import pickle
import logging


class Serialization:
    def __init__(self, result_path):
        self.result_path = result_path

    def dump_to_file(self, obj, filename):
        fullname = f"{self.result_path}/{filename}"
        logging.info(f"Dumping object of type {type(obj)} to {fullname}")
        with open(fullname, "wb") as outfile:
            pickle.dump(obj, outfile)

    def read_from_file(self, filename):
        fullname = f"{self.result_path}/{filename}"
        logging.info(f"Reading {fullname}")
        with open(fullname, "rb") as infile:
            return pickle.load(infile)
