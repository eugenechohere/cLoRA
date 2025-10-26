import polars as pl

if __name__ == "__main__":
    df = pl.read_parquet('hf://datasets/leonli66/nemotron-sft-general/nonthinking/train-00000-of-00447.parquet').to_pandas()
    df = df[~df["is_multiturn"]]


    df["prompt"] = df["prompt"].apply(lambda x: x[0]["content"])
    df["completion"] = df["target"].apply(lambda x: x)

    df["num_tokens"] = df.apply(lambda row: len(row["prompt"]+ row["completion"]) / 4, axis=1)
    df = df[df["num_tokens"]<1024]
    df[["prompt", "completion"]].to_json("/home/ubuntu/calhacks-continual-learning/infra/data/nemotron.jsonl",lines=True,orient="records")



