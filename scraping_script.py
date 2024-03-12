import os
import yaml
import re
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import (
    HTTPError,
    ConnectionError,
    RequestException,
    Timeout
)

import pandas as pd
import pyarrow
from pandas import DataFrame
from logger_file import logger

import pyspark
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T, Row
from pyspark.sql.window import Window
from pyspark import SparkContext

logger.info(f"Logger initialized successfully in file")

spark = SparkSession.builder.appName("Sec-Gov-Scraping").getOrCreate()
sc = SparkContext.getOrCreate()


def save_dataframepqt(df: DataFrame, path: str):
    df = df.coalesce(1)
    file_exists = os.path.exists(f"{path}")
    if file_exists:
        df.write.mode("append").parquet(path)
    else:
        df.write.mode("overwrite").parquet(path)

def get_request(client, url, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            res = client.get(url, params = {'render_js': 'False',})
            status = res.raise_for_status()
            if status == 200:
                data = res.content
                soup = BeautifulSoup(data, "html.parser")
                logger.info(f"Successfully extracted the soup from {url}")
                return soup
            else: 
                return None
        except HTTPError as errh:
            logger.error("Http Error:", errh)
            time.sleep(2)
        except ConnectionError as errc:
            logger.error("Error Connecting:", errc)
            time.sleep(2)
        except Timeout as errt:
            logger.error("Timeout Error:", errt)
            time.sleep(2)
        except RequestException as err:
            logger.error("OOps: Something Else", err)
            time.sleep(2)
        retries += 1
        time.sleep(2)
    return None

def scrape_document(cik_name: str, date: str, cik_num: str, accsNum: str, document: str, client, max_retries=3):
    retries = 0
    page_content = []
    while retries < max_retries:
        try:
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accsNum}/{document}"
            logger.info(f"Currently beginning scraping for cik name {cik_name} and document {document}")
            soup = get_request(client, url)
            logger.info(f"Completed scraping for cik name {cik_name} and document {document}")
            page_content.append({
                        "cik_name": cik_name,
                        "reporting_date": date,
                        "url":url,
                        "contents": soup
                    })
            return page_content
        except requests.exceptions.HTTPError as errh:
            logger.info("Http Error:", errh)
            time.sleep(2)
        except requests.exceptions.ConnectionError as errc:
            logger.info("Error Connecting:", errc)
            time.sleep(2)
        except requests.exceptions.Timeout as errt:
            logger.info("Timeout Error:", errt)
            time.sleep(2)
        except requests.exceptions.RequestException as err:
            logger.info("OOps: Something Else", err)
        retries += 1
        time.sleep(5)
    return None


def scrap_table(dict_records: dict):
    cik_num = dict_records["cik_number"]
    date = dict_records["reportDate"]
    accsNum = dict_records["accessionNumber"]
    document = dict_records["primaryDocument"]
    cik_name = dict_records["cik_name"]
    tables = scrape_document(cik_name, date, cik_num, accsNum, document, client)
    data = []
    for table in tables:
        data.append(Row(**table))
    return data



def process_chunked_df(processed_rows):
    if len(processed_rows) > 0:
        logger.info(len(processed_rows))
        table_data = spark.sparkContext.parallelize(processed_rows)
        table_df = spark.createDataFrame(table_data)
        table_df = table_df.select(
            F.col("_1.cik_name").alias("cik_name"),
            F.col("_1.reporting_date").alias("reporting_date"),
            F.col("_1.url").alias("url"),
            F.col("_1.contents").alias("contents"),
        )
        table_df = table_df.filter(F.col("contents").isNotNull())
        save_dataframepqt(table_df, "data/output/table_contents/new_table_contents")
        logger.info(
            f"Saved the document to 'data/output/table_contents/new_table_contents'"
        )


if __name__ == "__main__":
    api_key = ''
    client = ScrapingBeeClient(api_key=api_key)
    logger.info(f"Scraping Table content Started")
    #companies_df = spark.read.parquet("/home/kushal/IA-FEB7/data/output/companies_details/company_detail.parquet")
    companie1_df = spark.read.parquet(
        "data/output/companies_details/company_details_file"
    )
    companie2_df = spark.read.parquet(
        "data/output/companies_details/company_details_fil"
    )
    companies_df = companie1_df.union(companie2_df)
    companies_df = companies_df.withColumn(
        "row_id", F.row_number().over(Window.orderBy("cik_number"))
    )
    total_items = companies_df.count()
    batch_size = 110
    num_batches = (total_items + batch_size - 1) // batch_size
    logger.info(f"Total Number of Batches: {num_batches}")
    for i in range(0,num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_items)
        logger.info(f"Batche of index: {start_idx}-{end_idx} and {i}/{num_batches} and started.")
        chunked_df = companies_df.filter(
            (F.col("row_id") >= start_idx) & (F.col("row_id") < end_idx)
        )
        with ThreadPoolExecutor(max_workers=11) as executor:
            chunked_df = chunked_df.drop("row_id")
            pd_df = chunked_df.toPandas()
            processed_rows = list(executor.map(scrap_table, pd_df.to_dict(orient="records"))
            )
        logger.info(len(processed_rows))
        process_chunked_df(processed_rows)
    logger.info(f"Scraping Table content Completed")
