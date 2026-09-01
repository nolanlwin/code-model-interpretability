"""Audit the paper's prose against the academic-paper skill's quality rules.

Implements the checkable parts of
.claude/skills/academic-paper/references/writing_quality_check.md: flagged
high-frequency terms, em-dash and semicolon limits, throat-clearing openers,
binary-contrast repetition, and sentence-length variation.

The first run over paper/lp4fm_short/main.tex found 20 em dashes against a limit of 3,
14 semicolons against 4, and 11 repetitions of one contrast construction. None
of that is visible while writing; it needs counting. Run it after editing prose:

    uv run python scripts/writing_quality_check.py
"""
import re, pathlib, statistics as st
tex = pathlib.Path("paper/lp4fm_short/main.tex").read_text()
# strip preamble, comments, and math so prose counts are honest
body = tex.split("\\begin{document}",1)[1].split("\\appendix")[0]
body = re.sub(r"(?m)^%.*$", "", body)
body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body, flags=re.S)
prose = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)
prose = re.sub(r"\$[^$]*\$", " NUM ", prose)
words = len(prose.split())
print(f"  main-text words (approx): {words}\n")

em = body.count("---")
semi = prose.count(";")
print(f"  B. em dashes (---)      : {em}   [limit 3, recommend 0-1]  {'OVER' if em>3 else 'ok'}")
print(f"  B. semicolons           : {semi}   [limit {2*words//1000} at 2/1000w]  {'OVER' if semi>2*words/1000 else 'ok'}")

FLAG = ["delve","tapestry","landscape","pivotal","crucial","foster","showcase",
        "testament","navigate","leverage","realm","embark","underscore",
        "multifaceted","nuanced","comprehensive","robust","intricate",
        "cornerstone","paradigm","synergy","holistic","streamline",
        "cutting-edge","groundbreaking"]
hits = {t: len(re.findall(rf"\b{t}", prose, re.I)) for t in FLAG}
hits = {k:v for k,v in hits.items() if v}
print(f"\n  A. flagged terms        : {hits if hits else 'none'}")

THROAT = ["In the realm of","It's important to note","It is worth mentioning",
          "In today's","This serves as a testament","It goes without saying",
          "In order to","It should be noted","As a matter of fact",
          "When it comes to","At the end of the day","With that being said",
          "This section will","The following paragraph","We now turn"]
th = [t for t in THROAT if t.lower() in prose.lower()]
print(f"  C. throat-clearing      : {th if th else 'none'}")

# D. binary contrast
bc = re.findall(r"not [a-z ]{2,30}, (?:but )?(?:rather )?[a-z]", prose, re.I)
bc += re.findall(r"rather than", prose, re.I)
print(f"  D. binary contrast      : {len(bc)}   [limit 2]  {'OVER' if len(bc)>2 else 'ok'}")

# E. burstiness
sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose) if len(s.split())>3]
lens = [len(s.split()) for s in sents]
print(f"\n  E. sentences: {len(lens)}  mean {st.mean(lens):.1f}  sd {st.pstdev(lens):.1f}"
      f"  range {min(lens)}-{max(lens)}")
run=1; worst=1; wi=0
for i in range(1,len(lens)):
    if abs(lens[i]-lens[i-1])<=5: run+=1
    else:
        if run>worst: worst, wi = run, i-run
        run=1
print(f"     longest run of similar-length sentences: {worst} {'(flag: 5+)' if worst>=5 else ''}")
short = sum(1 for l in lens if l<=10)
print(f"     short sentences (<=10 words): {short} of {len(lens)}")
