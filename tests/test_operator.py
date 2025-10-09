import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import pytest
from basics.op_add import Add
from basics.op_mul import Multiply
from basics.op_exp import Exponential

def test_add_forward_backward():
	add_op = Add()
	x, y = 1, 4
	out = add_op(x, y)
	assert out == 5
	grads_x, grads_y = add_op.backward(1)
	assert grads_x == 1
	assert grads_y == 1

def test_multiply_forward_backward():
	mul_op = Multiply()
	x, y = 2, 3
	out = mul_op(x, y)
	assert out == 6
	grads_x, grads_y = mul_op.backward(2)
	assert grads_x == 2 * y
	assert grads_y == 2 * x

def test_exponential_forward_backward():
	exp_op = Exponential()
	x = 2
	out = exp_op(x)
	assert out == math.exp(2)
	grad = exp_op.backward(1)
	assert pytest.approx(grad, rel=1e-6) == math.exp(2)

def test_chain():
	# y = exp(add(mul(a, b), mul(c, d)))
	a, b, c, d = 2, 3, 2, 2
	multiply_op1 = Multiply()
	multiply_op2 = Multiply()
	add_op = Add()
	exp_op = Exponential()
	mul1 = multiply_op1(a, b)
	mul2 = multiply_op2(c, d)
	add = add_op(mul1, mul2)
	y = exp_op(add)
	assert y == math.exp(mul1 + mul2)

