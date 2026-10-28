"""Week 1: check GPU memory and that the Medusa checkpoint loads + decodes one prompt.

    python scripts/00_gpu_check.py                 # report only
    python scripts/00_gpu_check.py --load          # also load model and run one prompt
    python scripts/00_gpu_check.py --load --set model.load_in_4bit=true
"""

from _common import parse


def extra(ap):
    ap.add_argument("--load", action="store_true")


def main():
    args, cfg = parse(__doc__, extra)
    from mcal.model import gpu_report
    print(gpu_report())
    if not args.load:
        return

    import torch
    from mcal.collect import default_medusa_choices
    from mcal.data import vicuna_prompt
    from mcal.decode import StaticTree, medusa_greedy_generate, plain_greedy_generate
    from mcal.model import load_medusa

    model, tok = load_medusa(cfg)
    print(f"loaded {cfg['model']['name']} with H={model.medusa} heads; "
          f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    ids = tok([vicuna_prompt("Explain speculative decoding in two sentences.")], return_tensors="pt").input_ids
    out_g, sg = plain_greedy_generate(model, ids, 64)
    out_m, sm = medusa_greedy_generate(model, ids, StaticTree(default_medusa_choices()), 64)
    n = min(out_g.shape[1], out_m.shape[1])
    same = bool((out_g[0, :n].cpu() == out_m[0, :n].cpu()).all())
    print(f"greedy:  {sg.tokens_per_s:.1f} tok/s")
    print(f"medusa:  {sm.tokens_per_s:.1f} tok/s, {sm.mean_accepted:.2f} accepted tokens/step")
    print(f"outputs identical: {same}")
    print(tok.decode(out_m[0, ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
