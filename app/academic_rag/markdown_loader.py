"""
app/academic_rag/markdown_loader.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import logging
import os

from langchain_text_splitters import MarkdownHeaderTextSplitter
log = logging.getLogger(__name__)

def load_markdown_documents(folder_path):
    # Defining headers to track
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Initialize the splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    all_chunks = []

    if not os.path.exists(folder_path):
        return []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # This returns a list of LangChain Document objects
            segments = markdown_splitter.split_text(content)

            for segment in segments:
                # Add source metadata
                segment.metadata["file"] = file_name

                # Get Docs By domain
                segment.metadata["grade"] = extract_grade_from_filename(file_name)
                if segment.metadata["grade"].startswith("ef"):
                    segment.metadata["domain"] = "ef"
                elif segment.metadata["grade"].startswith("elcc"):
                    segment.metadata["domain"] = "elcc"
                else:
                    segment.metadata["domain"] = file_name.split('_')[0]

                log.info(f"File => {file_name} - {segment.metadata['domain']}")

                # Standardize 'title' for your existing logic
                # It takes the highest level header found in that chunk
                segment.metadata["title"] = (
                        segment.metadata.get("Header 1") or
                        segment.metadata.get("Header 2") or
                        segment.metadata.get("Header 3") or
                        file_name
                )
                all_chunks.append(segment)

    return all_chunks

def extract_grade_from_filename(filename):
    return filename.replace(".md", "").lower()