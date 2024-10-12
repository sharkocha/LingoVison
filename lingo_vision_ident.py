import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from openai import OpenAI
from datetime import datetime
import time
import os
import json

import create_index

index, docs = create_index.load_index()

start_time = time.time()

# prompt
WithPrompt = True
# RAG
WithRAG = True
# Servcie
WithProtocol = True
WithService = True
# TLS
WithTLS = True
# Banner
WithBanner = True
# Title
WithTitle = True

Start = 36
End = 333
RagNum = 3

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"result/ident_res_{current_time}.txt"

df = pd.read_excel("./dataset/unknown_dataset.xlsx")
# df = df.sample(frac=1, random_state=42).reset_index(drop=True) # randomization
features = df[
    ["IP", "Port", "Protocol", "Service", "Subject.CN", "Subject.O", "Title", "Banner"]
]#.iloc[Start:]
# true_labels = df["label"]#.iloc[Start:End]
device_types = [
    "router",
    "nas",
    "webcam",
    "firewall",
    "voip adapter",
    "gateway",
    "printer",
    "wap",
    "vpn",
    "load balancer",
    "proxy server",
    "ics",
    "media device",
    "mail server",
]


def query_llamafile(prompt) -> str:
    client = OpenAI(
        base_url="http://localhost:8081/v1",  # "http://<Your api-server IP>:port"
        api_key="sk-no-key-required",
    )
    completion = client.chat.completions.create(
        model="LLaMA_CPP",
        messages=[
            {
                "role": "system",
                "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return str(completion.choices[0].message.content)


def build_prompt(device_info, device_types) -> str:
    device_types_str = ", ".join([f'"{device}"' for device in device_types])
    subject_cn = "" if pd.isna(device_info["Subject.CN"]) else device_info["Subject.CN"]
    subject_o = "" if pd.isna(device_info["Subject.O"]) else device_info["Subject.O"]
    title = "" if pd.isna(device_info["Title"]) else device_info["Title"]

    feature_info = ""
    if WithProtocol:
        feature_info += f"Protocol: {device_info['Protocol']}\n"
    if WithService:
        feature_info += f"Service: {device_info['Service']}\n"
    if WithTLS:
        feature_info += f"Subject.CN: {subject_cn}\n"
        feature_info += f"Subject.O: {subject_o}\n"
    if WithTitle:
        feature_info += f"Title: {title}\n"
    if WithBanner:
        feature_info += f"Banner: {device_info['Banner']}\n"

    # prompt = f"We have several device types: {device_types_str}.\n"
    prompt = "We have several device types as following:\n"

    if WithPrompt:
        # with prompt
        prompt += '"router": a router or switch.\n'
        prompt += '"nas": a device for network attached storage.\n'
        prompt += '"webcam": a network camera.\n'
        prompt += '"firewall"\n'
        prompt += '"voip adapter": a device about VoIP.\n'
        prompt += '"gateway"\n'
        prompt += '"printer"\n'
        prompt += '"wap": a wireless router or access point for wireless access protocol(WLAN).\n'
        prompt += '"vpn": a VPN server/client or login point.\n'
        prompt += '"load balancer"\n'
        prompt += '"proxy server": a device for HTTP proxy.\n'
        prompt += '"ics": a device about Industrial Control Systems.\n'
        prompt += '"media device": a device designed to play, stream, or handle multimedia content such as audio, video, and images.\n'
        prompt += '"mail server"\n\n'
    else:
        # without prompt
        prompt += '"router"\n'
        prompt += '"nas"\n'
        prompt += '"webcam"\n'
        prompt += '"firewall"\n'
        prompt += '"voip adapter"\n'
        prompt += '"gateway"\n'
        prompt += '"printer"\n'
        prompt += '"wap"\n'
        prompt += '"vpn"\n'
        prompt += '"load balancer"\n'
        prompt += '"proxy server"\n'
        prompt += '"ics"\n'
        prompt += '"media device"\n'
        prompt += '"mail server"\n\n'

    if WithRAG:
        # RAG part
        emb = create_index.embed(feature_info)
        scores, doc_indices = index.search(emb, RagNum)
        search_results = [docs[ix] for ix in doc_indices[0]]
        prompt += "And here are some background knowledges that maybe useful for device type inferring:\n"
        prompt += "\n".join(search_results)
        # create_index.pprint_search_results(scores, doc_indices, docs)

    prompt += "\n\n"
    prompt += "Please infer the type of a device by infomations from one of its open port as following(Some info is vacant):\n"
    prompt += "-----------------------------------\n"
    prompt += feature_info
    prompt += "-----------------------------------\n\n"
    prompt += """
I hope you give me the device type and relative info that you infered in following JSON format and do not say any other words(Some JSON fields can be NULL if you don't know):

```
{
    "device type": "XXX",
    "confidence score": XX,
    "device productor": "XXX",
    "device version": "XXX",
    "affiliated organizations": "XXX,
    "inference reason": "XXX"
}
```

The device type you infered could be one of the types that I gave you or any other type if you think it's more correct and accurate. However, device type can be `unknown` if you cannot infer it at all.
"""

    return prompt


with open(output_filename, "w") as f:
    for idx, row in features.iterrows():
        prompt = build_prompt(row, device_types)
        prediction = query_llamafile(prompt)

        print(f"NO.{idx}============================================")
        # print(prompt)
        print(f"probe result:\n{prediction}\n")
        f.write(f"NO.{idx}============================================\n")
        f.write(f"{prediction}\n")


end_time = time.time()
execution_time = end_time - start_time
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)
print(
    f"executing time: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds"
)
