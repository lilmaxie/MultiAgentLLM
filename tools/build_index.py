# tools/build_index.py
import argparse, json
from utils.affina_markdown import process_markdown_file
from retriever.vector_store import VectorStore

p = argparse.ArgumentParser()
p.add_argument("dataset", help="Thư mục chứa .md (vd. data_campaign)")
args = p.parse_args()

vs = VectorStore(args.dataset, auto_build=False)   # chỉ load rỗng
nodes = process_markdown_file(f"{args.dataset}/Campaign.md")  # hoặc glob toàn thư mục
vs.build_index(nodes)
vs.save()
print("✔ Vector index saved to", vs.index_dir)
