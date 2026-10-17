import torch
import torch.utils.data

import cvtda.dumping
import cvtda.utils
import cvtda.zigzag.nn


def train_validate(
    model: torch.nn.Module,
    train_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    dumper: cvtda.dumping.UniversalDumper,
    epochs: int = 10,
    learning_rate: float = 1e-3,
):
    cvtda.utils.set_random_seed(42)
    dumper.execute(
        cvtda.zigzag.nn.train,
        "train_model_history",
        model,
        torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=3),
        torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=3),
        epochs=epochs,
        learning_rate=learning_rate,
    )
    if not dumper.has_dump("trained_model"):
        dumper.save_dump(model, "trained_model")


def validate_pretrained(
    model: torch.nn.Module,
    train_ds: torch.utils.data.Dataset,
    train_y: torch.Tensor,
    test_ds: torch.utils.data.Dataset,
    test_y: torch.Tensor,
    dumper: cvtda.dumping.UniversalDumper,
):
    cvtda.utils.set_random_seed(42)

    train_embeddings = dumper.execute(
        cvtda.zigzag.nn.precompute_embeddings,
        "train_embeddings",
        model,
        torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=3),
    )
    test_embeddings = dumper.execute(
        cvtda.zigzag.nn.precompute_embeddings,
        "test_embeddings",
        model,
        torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=3),
    )

    train_emb_ds = torch.utils.data.TensorDataset(train_embeddings, train_y)
    test_emb_ds = torch.utils.data.TensorDataset(test_embeddings, test_y)

    head = torch.nn.Sequential(torch.nn.LazyLinear(len(torch.unique(train_y))), torch.nn.Softmax(dim=1))
    dumper.execute(
        cvtda.zigzag.nn.train,
        "train_head_history",
        head,
        torch.utils.data.DataLoader(train_emb_ds, batch_size=128, shuffle=True, num_workers=3),
        torch.utils.data.DataLoader(test_emb_ds, batch_size=128, shuffle=False, num_workers=3),
    )
    if not dumper.has_dump("trained_head"):
        dumper.save_dump(head, "trained_head")
