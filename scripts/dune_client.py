#!/usr/bin/env python3

import os

import dotenv

from dune_client.client import DuneClient
from dune_client.query import Query
from dune_client.types import QueryParameter

query = Query(
    name="Sample Query",
    query_id=1215383,
    params=[
        QueryParameter.text_type(name="TextField", value="Word"),
        QueryParameter.number_type(name="NumberField", value=3.1415926535),
        QueryParameter.date_type(name="DateField", value="2022-05-04 00:00:00")
    ],
)
print("Results available at", query.url())

dotenv.load_dotenv()
dune = DuneClient(os.environ["DUNE_API_KEY"])
results = dune.refresh(query)
print(results.get_rows())
