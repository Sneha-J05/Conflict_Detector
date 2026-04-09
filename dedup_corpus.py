import json

def deduplicate_corpus():
    with open('data/corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    seen = set()
    deduped = []
    for art in corpus:
        aid = art['id']
        if aid not in seen:
            seen.add(aid)
            deduped.append(art)
            
    with open('data/corpus.json', 'w', encoding='utf-8') as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
        
    print(f"Removed {len(corpus) - len(deduped)} duplicates. {len(deduped)} articles remain.")

if __name__ == "__main__":
    deduplicate_corpus()
