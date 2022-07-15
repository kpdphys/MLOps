#!/usr/bin/env python
# coding: utf-8

import os
import json
from typing import Dict, NamedTuple
import logging
import random
import datetime
import argparse
from collections import namedtuple

from fastparquet import ParquetFile
import kafka


class RecordMetadata(NamedTuple):
    topic: str
    partition: int
    offset: int


def get_filenames(root: str) -> list:
    parquet_filenames = []

    for path, _, filenames in os.walk(root):
        parquet_filenames.extend(
            os.path.join(path, filename) \
            for filename in filenames \
            if filename.endswith(".parquet")
        )
    return parquet_filenames

def get_dataframe(filenames: list):
    return (ParquetFile(filename).to_pandas() for filename in filenames)

def clear_data(dataframe):
    cols = ["instanceId_objectType",
            "audit_clientType",
            "audit_resourceType",
            "metadata_numSymbols",
            "auditweights_numLikes",
            "auditweights_numDislikes",
            "auditweights_ctr_negative",
            "metadata_numPhotos",
           ]
    return dataframe[cols]

def get_json(dataframe_rows) -> str:
    return (ind_series for ind_series in dataframe_rows)

def serialize(msg: str) -> bytes:
    return msg.encode("utf-8")

def send_message(producer: kafka.KafkaProducer, 
                 topic: str,
                 key: int,
                 data: str) -> RecordMetadata:
    future = producer.send(
        topic=topic,
        key=str(key).encode("ascii"),
        value=data,
    )

    # Block for 'synchronous' sends
    record_metadata = future.get(timeout=10)
    return RecordMetadata(
        topic=record_metadata.topic,
        partition=record_metadata.partition,
        offset=record_metadata.offset,
    )

#def get_dataframe(dataset_path: list):
#    filenames = get_filenames(dataset_path)
#    return (dataframe for dataframe in get_dataframe(filenames))

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d",
        "--dataset_path",
        default="./dataset/test",
        help="path to dir with dataset",
    )
    argparser.add_argument(
        "-b",
        "--bootstrap_server",
        default="localhost:9092",
        help="kafka server address:port",
    )
    argparser.add_argument(
        "-t", "--topic", default="sna_data", help="kafka topic to consume"
    )

    args = argparser.parse_args()

    print(args.bootstrap_server)
    print(args.dataset_path)
    print(args.topic)


    producer = kafka.KafkaProducer(
        bootstrap_servers=[args.bootstrap_server],
        api_version=(0, 10, 2),
        value_serializer=serialize,
    )

    try:
        #for dataframe in get_dataframe(args.dataset_path):
        filenames = get_filenames(args.dataset_path)
        for dataframe in get_dataframe(filenames):
            dataframe = clear_data(dataframe)
            for key, row in dataframe.iterrows():
                json = row.to_json()
                record_md = send_message(producer, args.topic, key, json)
                if (record_md.offset % 100 == 0):
                    print(
                        f"Msg sent. Topic: {record_md.topic}, partition:{record_md.partition}, offset:{record_md.offset}"
                    )
    except kafka.errors.KafkaError as err:
        logging.error(err)
    producer.close()

if __name__ == "__main__":
    main()
