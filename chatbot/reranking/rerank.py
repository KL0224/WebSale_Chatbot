from FlagEmbedding import FlagReranker

class Reranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        self.model_name = model_name
        self.reranker = FlagReranker(self.model_name, use_fp16=True)

    def rerank(self, query, candidates, top_n=5):
        if not candidates:
            return []

        pairs = []
        for candidate in candidates:
            payload = candidate.payload
            context = payload.get("text_answer", "")
            pairs.append([query, context])

        scores = self.reranker.compute_score(pairs, normalize=True)

        # Wrapper structure
        reranked = []
        for i, candidate in enumerate(candidates):
            reranked.append({
                "point": candidate,
                "rerank_score": float(scores[i])
            })

        # Sort theo rerank score
        reranked_sorted = sorted(
            reranked,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_sorted[:top_n]
