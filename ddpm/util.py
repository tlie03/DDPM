import torch


def sinusoidal_encoding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Creates a sinusoidal encoding for every entry in t with dimension dim.
    :param t: t values that should be encoded
    :param dim: length of the encoded vectors. Should be dividable by 2 otherwise,
    will be rounded to the next lower integer dividable by 2.
    :return: torch tensor of shape (t.shape, 2 * (dim // 2))
    """
    half = dim // 2
    nominator = t.unsqueeze(-1).repeat(*([1] * len(t.shape)), half)
    denominator = torch.pow(10000, (2 * torch.arange(half) / dim))
    raw_values = nominator / denominator
    return torch.cat([torch.sin(raw_values), torch.cos(raw_values)], dim=-1)
