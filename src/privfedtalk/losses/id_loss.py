def identity_cosine_loss(id_pred, id_ref):
    return 1.0 - (id_pred * id_ref).sum(dim=-1).mean()
