import torch
from torch import nn


class MLP(nn.Module):
    """The MLP of the ClimSim paper (Yu, Hu et al. 2024, SI section 3).

    `floor` is the smallest value of each output.  The paper applies a ReLU to the eight surface outputs, because
    these fluxes and rates are not negative.  With standardised targets, zero in physical units is not zero for the
    network.  Thus the floor of a surface output is the standardised value of zero.  The other outputs have no floor.
    """

    def __init__(self, n_in: int, n_out: int, hidden: tuple = (768, 640, 512, 640, 640, 128), negative_slope: float = 0.15,
                 floor: torch.Tensor | None = None):
        super().__init__()
        widths = [n_in, *hidden]
        self.body = nn.Sequential(*[m for a, b in zip(widths[:-1], widths[1:])
                                    for m in (nn.Linear(a, b), nn.LeakyReLU(negative_slope))])
        self.head = nn.Linear(widths[-1], n_out)
        self.register_buffer("floor", floor if floor is not None else torch.full((n_out,), -torch.inf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(self.head(self.body(x)), self.floor)
