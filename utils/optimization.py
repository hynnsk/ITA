"""AdamW parameter groups for ITA's trainable parameters."""


def prep_optim_params_groups(args, model):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # Preserve the weight-decay grouping of the original training recipe.
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    return [
        {
            "params": [p for n, p in parameters if not any(s in n for s in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in parameters if any(s in n for s in no_decay)],
            "weight_decay": 0.0,
        },
    ]
