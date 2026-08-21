"""Emit BibTeX from live Crossref/arXiv records, never from memory."""
import json, subprocess, sys, urllib.parse, re

DOIS = {
 "hewitt2019control":   "10.18653/v1/d19-1275",
 "sajaniemi2002roles":  "10.1109/hcc.2002.1046340",
 "karmakar2021what":    "10.1109/ase51524.2021.9678927",
 "troshin2022probing":  "10.18653/v1/2022.blackboxnlp-1.31",
 "libovicky2020language":"10.18653/v1/2020.findings-emnlp.150",
 "voita2020mdl":        "10.18653/v1/2020.emnlp-main.14",
 "belinkov2022probing": "10.1162/coli_a_00422",
 "littell2017uriel":    "10.18653/v1/e17-2002",
 "lin2019choosing":     "10.18653/v1/p19-1301",
}
ARXIV = {
 "zhu2022xlcost":  "2206.08474",
 "hui2024qwencoder":"2409.12186",
 "qwen2024qwen25": "2412.15115",
}

def curl(url):
    return subprocess.run(["curl","-sL","--max-time","30",url,
        "-H","User-Agent: mailto:naingoolwin.astrio@gmail.com"],
        capture_output=True, text=True).stdout

def esc(s):
    return (s.replace("&","\\&").replace("#","\\#").replace("_","\\_"))

out=[]
for key, doi in DOIS.items():
    d = json.loads(curl(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))["message"]
    au = " and ".join(f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                      for a in d.get("author", []))
    title = esc(" ".join(d["title"][0].split()))
    venue = esc(" ".join((d.get("container-title") or ["?"])[0].split()))
    # Some older IEEE records carry no issued date. Falling back to `created`
    # yields the indexing date, which for Sajaniemi (HCC 2002) reads 2003.
    # Prefer a year stated in the venue or event name; index date last.
    year = d["issued"]["date-parts"][0][0]
    if not year:
        hay = " ".join([(d.get("container-title") or [""])[0],
                        (d.get("event") or {}).get("name", "")])
        m = re.search(r"\b(?:19|20)\d{2}\b", hay)
        year = m.group(0) if m else d.get("created", {}).get("date-parts", [[None]])[0][0]
    typ = "article" if d.get("type") == "journal-article" else "inproceedings"
    field = "journal" if typ == "article" else "booktitle"
    out.append(f"@{typ}{{{key},\n  title     = {{{title}}},\n  author    = {{{au}}},\n"
               f"  {field:<9} = {{{venue}}},\n  year      = {{{year}}},\n  doi       = {{{doi}}}\n}}")

for key, aid in ARXIV.items():
    x = curl(f"https://export.arxiv.org/api/query?id_list={aid}")
    e = re.search(r"<entry>(.*?)</entry>", x, re.S).group(1)
    title = esc(" ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split()))
    names = [n for n in re.findall(r"<name>([^<]+)</name>", e) if n.strip() not in (":",)]
    au = " and ".join(names)
    year = re.search(r"<published>(\d{4})", e).group(1)
    out.append(f"@article{{{key},\n  title   = {{{title}}},\n  author  = {{{au}}},\n"
               f"  journal = {{arXiv preprint arXiv:{aid}}},\n  year    = {{{year}}},\n"
               f"  eprint  = {{{aid}}},\n  archivePrefix = {{arXiv}}\n}}")

print("% Every entry below was emitted from a live Crossref or arXiv record.\n"
      "% Regenerate with scripts/make_bib.py; do not hand-edit fields.\n")
print("\n\n".join(out))
