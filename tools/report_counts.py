import glob, os
stubs = glob.glob("data/processed/**/*-pdf.md", recursive=True)
pdfs  = glob.glob("data/raw/**/*.pdf", recursive=True)
mds   = glob.glob("data/processed/**/*.md", recursive=True)
print("PDF stubs (-pdf.md):", len(stubs))
print("Raw PDFs          :", len(pdfs))
print("Markdown files    :", len(mds))
print("Examples (stubs):", stubs[:3])
print("Examples (pdfs):", pdfs[:3])
