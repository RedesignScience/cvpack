"""
.. class:: MultilayerPerceptron
   :platform: Linux, MacOS, Windows
   :synopsis: A feed-forward neural network having simple CVs as inputs

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import functools
import itertools
from typing import Iterable, Sequence

import numpy as np
import openmm

import sympy
from numpy.typing import ArrayLike

from .cvpack import AbstractCollectiveVariable

from sympy.abc import x


class MultilayerPerceptron(openmm.CustomCVForce, AbstractCollectiveVariable):
    """
    The single output of a feed-forward, fully connected neural network with :math:`N` inputs and
    :math:`n+1` layers:

    .. math::

        f_{\\rm MLP}({\\bf r}) = f_n({\\bf r})

    where

    .. math::

        {\\bf f}_i({\\bf r}) = g\\Big({\\bf W}_i^T {\\bf f}_{i-1}({\\bf r}) + {\\bf b}_i\\Big)
            \\qquad i = 1, \\ldots, n

    and

    .. math::

        {\\bf f}_0({\\bf r}) = \\begin{bmatrix}
            t_1(y_1({\\bf r})) \\\\
            \\vdots \\\\
            t_N(y_N({\\bf r}))
        \\end{bmatrix}

    In the equations above, :math:`{\\bf f}_i({\\bf r})` is the output of the i-th layer,
    :math:`y_1({\\bf r}), \\ldots, y_N({\\bf r})` are the collective variables fed to the network,
    :math:`t_1(x), \\ldots, t_N(x)` are transforms applied to these collective variables,
    :math:`{\\bf W}_i` and :math:`{\\bf b}_i` are the weight matrix and bias vector of the i-th
    layer, respectively, and :math:`g(x)` is the activation function, applicable element-wise to a
    vector of any size.

    The weight matrices must be conformable with respect to the product
    :math:`{\\bf W}_1 {\\bf W}_2 \\cdots {\\bf W}_n`, whose result must be a column vector equal in
    size to the number of neurons in the input layer. Each bias vector must be sized so that the
    product :math:`{\\bf W}_i {\\bf b}_i` is also possible.

    Parameters
    ----------
        collective_variables
            The collective variables to be used as inputs
        weight_matrices
            The weight matrices of the hidden layers
        bias_vectors
            The bias vectors of the hidden layers
        activation_function
            The activation function to be used. It must be a function of a single variable named
            ``x``. Defaults to ``x*erf(x)``, the Gaussian error linear unit (GELU).
        transforms
            The transforms to be applied to the collective variables. Each transform must be a
            function of a single variable named ``x``. Defaults to ``x`` for all collective
            variables.

    Examples
    --------
        >>> from cvpack import MultilayerPerceptron, Distance, Angle, Torsion
        >>> import numpy as np
        >>> from openmm import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> collective_variables = [
        ...     Distance(0, 1),
        ...     Distance(1, 2),
        ...     Distance(2, 3),
        ...     Angle(0, 1, 2),
        ...     Angle(1, 2, 3),
        ...     Torsion(0, 1, 2, 3),
        ... ]
        >>> mlp = MultilayerPerceptron(
        ...     collective_variables,
        ...     [np.ones((6, 3)), np.ones((3, 3)), np.ones((3, 1))],
        ...     [np.ones(3), np.ones(3), np.ones(1)],
        ...     transforms = ["1/x", "1/x", "x", "sin(x)", "sin(x)", "cos(x)"],
        ... )

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        collective_variables: Sequence[openmm.Force],
        weight_matrices: Sequence[ArrayLike],
        bias_vectors: Sequence[ArrayLike],
        activation_function: str = "x*erf(x)",
        transforms: Iterable[str] = itertools.repeat("x"),
    ) -> None:
        num_inputs = len(collective_variables)
        matrices = [np.array(matrix) for matrix in weight_matrices]
        vectors = [np.array(vector).ravel() for vector in bias_vectors]
        input_data = list(zip(collective_variables, map(sympy.sympify, transforms)))
        activation = sympy.sympify(activation_function)

        # Sanity checks
        assert len(input_data) == num_inputs, "Wrong number of transformations"
        assert len(vectors) == len(matrices), "Wrong number of weight matrices or bias vectors"
        try:
            product = functools.reduce(np.dot, matrices)
        except ValueError as error:
            raise ValueError("Weight matrices do not conform") from error
        assert product.shape == (
            num_inputs,
            1,
        ), "Weight matrices incompatible with number of provided CVs or with a single output"
        assert all(
            matrix.shape[1] == len(vector) for matrix, vector in zip(matrices, vectors)
        ), "Bias vectors and weight matrices do not conform"
        assert activation.free_symbols == {x}, "Invalid activation function"
        assert all(data[1].free_symbols == {x} for data in input_data), "Invalid transform"
        super().__init__(activation_function)
