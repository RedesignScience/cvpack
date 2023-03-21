"""
.. class:: MultilayerPerceptron
   :platform: Linux, MacOS, Windows
   :synopsis: A feed-forward neural network having simple CVs as inputs

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import openmm
from numpy.typing import ArrayLike

from .cvpack import AbstractCollectiveVariable


class MultilayerPerceptron(openmm.CustomCVForce, AbstractCollectiveVariable):
    """
    A feed-forward neural network with single output, having other collective variables as inputs:

    .. math::

        f({\\bf r}) = a\\Big({\\bf w}^T {\\bf f}_n({\\bf r}) + b\\Big)

    where :math:`n` is the number of hidden layers in the network, :math:`{\\bf w}` is the vector
    of weights of the output layer, :math:`b` is the bias of the output layer, :math:`{\\bf f}_n`
    is the vector of outputs of the last hidden layer, and :math:`a` is the activation function.
    The size of :math:`{\\bf w}` is equal to the number of neurons in the n-th hidden layer.

    The output of the i-th hidden layer is computed as:

    .. math::

        {\\bf f}_i({\\bf r}) = a\\Big({\\bf W}_i {\\bf f}_{i-1}({\\bf r}) + {\\bf b}_i\\Big)

    where :math:`{\\bf W}_i` and :math:`{\\bf b}_i` are the weight matrix and bias vector of the
    i-th hidden layer, respectively. The activation function is applied element-wise to a vector
    of any size. The number of rows of :math:`{\\bf W}_i` and the size of :math:`{\\bf b}_i` are
    equal to the number of neurons in the i-th hidden layer, while the number of columns of
    :math:`{\\bf W}_i` is equal to the number of neurons in the previous layer. The vector of inputs
    of the first hidden layer, :math:`{\\bf f}_0({\\bf r})`, is a set of transformed collective
    variables.

    Parameters
    ----------
        collective_variables
            The collective variables to be used as inputs
        transformations
            The transformations to be applied to the collective variables. Each transformation must
            be a function of a single variable named ``x``, which will be replaced by the
            corresponding collective variable.
        output_weights
            The weights of the output layer
        output_bias
            The bias of the output layer
        weight_matrices
            The weight matrices of the hidden layers
        bias_vectors
            The bias vectors of the hidden layers
        activation_function
            The activation function to be used. It must be a function of a single variable named
            ``x``. Defaults to ``x*erf(x)``, the Gaussian error linear unit (GELU).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        collective_variables: Sequence[openmm.Force],
        transformations: Sequence[str],
        output_weights: Sequence[float],
        output_bias: float,
        weight_matrices: Sequence[ArrayLike],
        bias_vectors: Sequence[ArrayLike],
        activation_function: str = "x*erf(x)",
    ) -> None:
        num_inputs = len(collective_variables)
        num_hidden_layers = len(weight_matrices)
        assert len(transformations) == num_inputs, "Wrong number of transformations"
        assert len(bias_vectors) == num_hidden_layers, "Wrong number of bias vectors"
        matrices = [np.reshape(output_weights, (-1, 1))] + list(map(np.array, weight_matrices))
        vectors = [np.reshape(output_bias, (1, 1))] + list(map(np.array, bias_vectors))
        for i, (matrix, vector) in enumerate(zip(matrices, vectors)):
            if i > 0:
                assert matrices[i - 1].shape[1] == matrix.shape[0], "Weight matrices do not conform"
            assert matrix.shape[0] == len(vector), "Wrong shape of bias vector"
        super().__init__(activation_function)
