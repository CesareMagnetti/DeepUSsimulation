import torch.nn as nn
from torch.nn import functional as F
import torch

class MSELoss(nn.MSELoss):

    def getLossName(self):
        return 'MSELoss'

class MaskedMSELoss(nn.MSELoss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input `x` and target `y`, augmented with target > threshold mask

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size\_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size\_average} = \text{False}.
        \end{cases}

    The target is masked using:

    .. math::

        mask = (target > threshold).type(target.type())


    The mean square error loss is only evaluated on pixels that are contained in the mask. The output MSE loss
    is averaged on the batch dimension

    Args:
        reduction is fixed to 'mean' for in-house purposes
        input (array-like): The input tensor
        target (array-like): The target tensor
        threshold (floating point number): Threshold applied to the target under which the loss is not calculated

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = MaskedMSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, reduction='mean', threshold=0):
        super(nn.MSELoss, self).__init__(size_average=None, reduce=None, reduction=reduction)
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, input, target):

        #mask = (input>self.threshold)*(target > self.threshold)
        mask = target > self.threshold
        input_masked = torch.masked_select(input, mask)
        target_masked = torch.masked_select(target, mask)
        out = F.mse_loss(input_masked,
                         target_masked,
                         reduction='sum')
        if self.reduction == 'sum':
            out.div_(input.size(0))  # @todo This divides by batch size only so effectively there is a sum
            pass
        elif self.reduction == 'mean':
            if input_masked.size(0)>0:
                out.div_(input_masked.size(0)) # @todo Check this with Nico, I think it was a bug. Maybe this is related to the batch size? Before, it divided by the batch size only
            #else:
            #    out = torch.zeros_like(out) # if there is no overlap we return 0. This is not great because the network pulls images appart.
        else:
            raise NotImplementedError('reduction should be sum | mean. Unknown value: {}'.format(self.reduction))
        return out

    def getLossName(self):
        return 'MaskedMSELoss'