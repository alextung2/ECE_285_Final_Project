"""Untimed comparison of teacher-forced TraceSet labels with tree verification."""
import numpy as np
from .precision import ranked_probabilities


class TraceObserver:
    def __init__(self, traces, reference, prompt_length):
        self.traces, self.reference, self.prompt_length = traces, reference, prompt_length
        self.rows = []

    def __call__(self, ids, choices, buffers, candidates, logits, head_logits):
        import torch
        position = ids.shape[1] - self.prompt_length
        if position >= self.traces.n_steps:
            return  # incomplete H-token horizon is intentionally not labelled
        probs, tokens = ranked_probabilities(head_logits, self.traces.topk)
        same_prefix = torch.equal(ids, self.reference[:, :ids.shape[1]])
        online_tokens = tokens.cpu().numpy()
        offline_tokens = self.traces.topk_token[position]
        changed = np.argwhere(online_tokens != offline_tokens)
        used = {(depth, rank) for path in choices for depth, rank in enumerate(path)}
        rank_changes = [dict(head=int(h), rank=int(r), selected=(int(h), int(r)) in used,
                             offline_token=int(offline_tokens[h, r]), online_token=int(online_tokens[h, r]))
                        for h, r in changed]
        same_ranks = len(rank_changes) == 0
        max_prob_error = float(np.max(np.abs(probs.cpu().numpy() - self.traces.topk_prob[position])))
        matches = (candidates[:, 1:] == logits[:, :-1].argmax(-1)).cpu().numpy()
        paths = sorted(choices, key=lambda p: (len(p), p))
        retrieval = buffers['retrieve_indices'].cpu().tolist()
        mismatches = []
        for node, path in enumerate(paths, 1):
            row, depth = next((r, d) for r, rr in enumerate(retrieval)
                              for d, index in enumerate(rr) if index == node)
            online = bool(matches[row, :depth].all())
            offline = bool(np.array_equal(self.traces.gold_rank[position, :depth], path))
            if online != offline:
                mismatches.append(path)
        self.rows.append(dict(position=position, same_prefix=same_prefix, same_ranks=same_ranks,
                              rank_changes=rank_changes, nodes_checked=len(paths),
                              max_probability_error=max_prob_error, label_mismatches=mismatches))

    @property
    def passed(self):
        return bool(self.rows) and all(r['same_prefix'] and r['same_ranks']
                                      and not r['label_mismatches'] for r in self.rows)
